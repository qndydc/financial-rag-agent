import os
import sys
import json
import hashlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from configs import rag_config
from rag import load_vector_store, create_hybrid_retriever


def get_all_documents():
    cache_path = os.path.join(rag_config.VECTOR_STORE_DIR, "all_documents.json")

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"缓存文件不存在：{cache_path}")

    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in data
    ]


def get_chunk_id(doc: Document) -> str:
    metadata = doc.metadata

    if metadata.get("chunk_id"):
        return str(metadata["chunk_id"])

    if metadata.get("id"):
        return str(metadata["id"])

    source = metadata.get("source", "unknown_file")
    page_num = metadata.get("page_num", metadata.get("page", "unknown_page"))

    content_hash = hashlib.md5(
        doc.page_content.encode("utf-8")
    ).hexdigest()[:12]

    return f"{source}::page_{page_num}::chunk_{content_hash}"


def parse_user_selection(user_input: str, max_index: int):
    user_input = user_input.strip().lower()

    if user_input in ["q", "quit", "exit"]:
        return "quit"

    if user_input in ["s", "skip"]:
        return "skip"

    if user_input in ["", "0", "none", "no"]:
        return []

    selected = []

    if "," in user_input or " " in user_input:
        parts = user_input.replace(",", " ").split()
        for p in parts:
            if p.isdigit():
                selected.append(int(p))
    else:
        for ch in user_input:
            if ch.isdigit():
                selected.append(int(ch))

    selected = [i for i in selected if 1 <= i <= max_index]
    return list(dict.fromkeys(selected))


def build_retrieval_gt_dataset(
    queries,
    output_path="retrieval_gt_dataset.json",
    label_mode="manual",          # manual / auto
    auto_score_threshold=0.8,
    fallback_first=True,          # auto模式下没有超过阈值时，是否取第一个
    search_mode="hybrid",         # vector / bm25 / hybrid
    use_reranker=True,
    retriever_score_threshold=0.3,
    device="cuda",
    preview_chars=500
):
    """
    label_mode:
        manual：人工输入 12345 选择 GT
        auto：自动选择 score > auto_score_threshold 的 chunk_id
              如果没有超过阈值，且 fallback_first=True，则返回第一个 chunk_id
    """

    print("=" * 100)
    print("🔥 开始构建检索 GT 数据集")
    print(f"标注模式：{label_mode}")
    print(f"检索模式：{search_mode} | Reranker：{use_reranker}")
    print("=" * 100)

    print("🔹 加载向量库...")
    vs = load_vector_store(rag_config.VECTOR_STORE_DIR)

    print("🔹 加载缓存文档...")
    all_docs = get_all_documents()
    print(f"✅ 总文档数量：{len(all_docs)}")

    print("🔹 初始化检索器...")
    search = create_hybrid_retriever(
        vectorstore=vs,
        all_documents=all_docs,
        score_threshold=retriever_score_threshold,
        device=device
    )

    dataset = []

    for q_idx, query in enumerate(queries, start=1):
        print("\n" + "=" * 100)
        print(f"🔍 [{q_idx}/{len(queries)}] Query：{query}")
        print("=" * 100)

        try:
            docs = search(
                query=query,
                mode=search_mode,
                use_reranker=use_reranker
            )
        except Exception as e:
            print(f"❌ 检索失败：{e}")
            dataset.append({"query": query, "gt": []})
            continue

        if not docs:
            print("⚠️ 没有召回任何文档")
            dataset.append({"query": query, "gt": []})
            continue

        chunk_ids = []

        for i, doc in enumerate(docs, start=1):
            chunk_id = get_chunk_id(doc)
            chunk_ids.append(chunk_id)

            title = doc.metadata.get("title", "未知")

            score = doc.metadata.get("score", "N/A")
            content = doc.page_content.replace("\n", " ")[:preview_chars]

            print(f"\n【{i}】")
            print(f"chunk_id：{chunk_id}")
            print(f"题目：{title}")
            print(f"score：{score}")
            print(f"内容：{content}...")
            print("-" * 100)

        if label_mode == "auto":
            gt = []

            for doc, chunk_id in zip(docs, chunk_ids):
                score = doc.metadata.get("score", None)

                try:
                    score = float(score)
                except Exception:
                    score = None

                if score is not None and score > auto_score_threshold:
                    gt.append(chunk_id)

            if not gt and fallback_first:
                continue
                gt = [chunk_ids[0]]
                print(f"⚠️ 没有 score > {auto_score_threshold} 的结果，已 fallback 选择第 1 条")
            else:
                print(f"✅ 自动选择 score > {auto_score_threshold}")

        elif label_mode == "manual":
            print("\n请选择哪些 chunk 计入 GT：")
            print("  123    表示选择第1、2、3条")
            print("  1,3,5  表示选择第1、3、5条")
            print("  0      表示没有相关chunk")
            print("  s      跳过该问题")
            print("  q      保存并退出")

            user_input = input("你的选择：")
            selected = parse_user_selection(user_input, len(docs))

            if selected == "quit":
                print("🛑 用户选择退出，正在保存当前结果...")
                break

            if selected == "skip":
                print("⏭️ 已跳过该问题")
                continue

            gt = [chunk_ids[i - 1] for i in selected]

        else:
            raise ValueError("label_mode 只能是 'manual' 或 'auto'")

        gt = list(dict.fromkeys(gt))

        dataset.append({
            "query": query,
            "gt": gt
        })

        print(f"\n✅ 当前 query 写入 GT 数量：{len(gt)}")
        for cid in gt:
            print(f"   - {cid}")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("🎉 GT 数据集构建完成")
    print(f"📄 输出文件：{output_path}")
    print("=" * 100)

    return dataset


if __name__ == "__main__":
    """
    queries = [
        "海光信息2024年归属于上市公司股东的净利润是多少？",
        "中芯国际2024年的资本开支总额是多少？",
        "韦尔股份2024年图像传感器业务收入是多少？",
        "兆易创新2024年存储芯片产品收入是多少？",
        "澜起科技2024年内存接口芯片业务收入是多少？",

        "紫光国微2024年智能安全芯片业务收入是多少？",


        "科大讯飞2024年营业收入是多少？",


        "科大讯飞2024年教育业务收入是多少？",


        "宁德时代2024年动力电池系统收入是多少？",

        "隆基绿能2024年光伏组件出货量是多少？",
        "阳光电源2024年储能系统业务收入是多少？",
        "亿纬锂能2024年动力电池业务收入是多少？",


        "隆基绿能2024年综合毛利率是多少？",
        "宁德时代2024年前五大客户销售占比是多少？",

        "沈阳新松机器人自动化股份有限公司机器人及智能制造系统业务收入是多少？",
        "汇川技术2024年工业自动化控制产品收入是多少？",
        "绿的谐波2024年谐波减速器业务收入是多少？",
        "天奇自动化工程2024年人形机器人业务进展如何？",
        "机器人公司2024年机器人与智能制造业务收入是多少？",
        "天奇自动化工程2024年伺服系统业务收入是多少？",
        "美的集团2024年工业机器人销量是多少？",
        "绿的谐波2024年主要客户来自哪些行业？",
        "汇川技术2024年研发投入是多少？",

        "美的2024年核心本地商业收入是多少？",
        "兴业证券2024年营业收入是多少？",
        "鄂尔多斯资源2024年净利润是多少？",
        "中国联通2024年速运物流业务收入是多少？",
        "海南航空2024年客运收入是多少？",
        "宁百货大楼2024年零售业务收入是多少？",
        "北京首旅酒店2024年酒店数量是多少？",
        "上海艾为电子技术2024年毛利率是多少？",
        "同庆楼餐饮2024年新业务经营亏损是多少？"
        # ================== IT / 计算 / AI ==================
        "中科曙光2024年营业收入是多少？",
        "中科曙光2024年高性能计算业务收入占比是多少？",
        "中国移动2024年通信服务收入是多少？",
        "中国移动2024年5G用户数量是多少？",
        "东方财富信息股份有限公司2024年证券业务收入是多少？",
        "东方财富2024年净利润是多少？",
        "招商证券2024年投资银行业务收入是多少？",
        "方正证券2024年营业收入是多少？",

        # ================== 半导体 ==================
        "深圳市汇顶科技股份有限公司2024年芯片业务收入是多少？",
        "汇顶科技2024年研发投入是多少？",
        "紫光国芯微电子股份有限公司2024年智能安全芯片收入是多少？",
        "山东天岳先进科技股份有限公司2024年碳化硅材料收入是多少？",
        "江苏华海诚科新材料股份有限公司2024年电子材料收入是多少？",

        # ================== 新能源 ==================
        "江西赣锋锂业集团股份有限公司2024年锂盐产品收入是多少？",
        "赣锋锂业2024年电池业务收入占比是多少？",
        "蔚蓝锂芯2024年锂电池出货量是多少？",
        "天能电池集团股份有限公司2024年电池业务收入是多少？",
        "中节能风力发电股份有限公司2024年发电量是多少？",
        "浙江省新能源投资集团股份有限公司2024年新能源装机容量是多少？",

        # ================== 工业 / 自动化 / 装备 ==================
        "中控技术股份2024年工业自动化业务收入是多少？",
        "中控技术2024年研发投入占比是多少？",
        "湘潭电机股份有限公司2024年电机产品收入是多少？",
        "河南思维自动化设备股份有限公司2024年轨道交通业务收入是多少？",
        "西安陕鼓动力股份有限公司2024年能源装备业务收入是多少？",
        "上海华峰铝业股份有限公司2024年铝加工产品收入是多少？",
        "阜新德尔汽车部件股份有限公司2024年汽车零部件收入是多少？",
        "惠州市德赛西威汽车电子股份有限公司2024年智能座舱业务收入是多少？",

        # ================== 医疗 / 服务 ==================
        "卫宁健康科技集团股份有限公司2024年医疗信息化业务收入是多少？",
        "乐普（北京）医疗器械股份有限公司2024年医疗器械收入是多少？",
        "上海悦心健康集团股份有限公司2024年医疗服务收入是多少？",

        # ================== 消费 / 餐饮 ==================
        "中国全聚德(集团)股份有限公司2024年餐饮业务收入是多少？",

        # ================== AI / 具身智能 ==================
        "奥比中光科技集团股份有限公司2024年3D视觉产品收入是多少？",
    ]
    """
    queries = [
        # ================== 半导体 / 电子（对比） ==================
        "对比汇顶科技和紫光国芯2024年的研发投入规模和占比。",
        "汇顶科技与山东天岳在2024年的主营业务收入结构有何差异？",
        "紫光国芯和华海诚科在2024年的核心产品收入分别是多少？",
        "中科曙光和中国移动2024年的IT相关业务收入规模对比如何？",

        # ================== 新能源（对比） ==================
        "赣锋锂业和天能电池2024年电池相关业务收入占比有何差异？",
        "蔚蓝锂芯和赣锋锂业2024年的锂电池业务发展重点分别是什么？",
        "中节能风电与浙江新能源投资集团2024年装机规模对比如何？",
        "天能电池和蔚蓝锂芯2024年电池出货量或销量情况对比如何？",

        # ================== 工业 / 自动化（对比） ==================
        "中控技术与思维自动化2024年工业自动化业务收入对比如何？",
        "湘潭电机和陕鼓动力2024年装备制造业务收入规模差异如何？",
        "德赛西威与阜新德尔在2024年汽车电子业务上的定位有何不同？",
        "中控技术与汇川类企业在2024年的研发投入强度对比如何？",

        # ================== 金融 / 服务（对比） ==================
        "东方财富与招商证券2024年证券业务收入结构有何差异？",
        "招商证券与方正证券2024年投资银行业务收入对比如何？",
        "中国平安与东方财富2024年金融科技业务收入情况有何不同？",

        # ================== 医疗 / 服务（对比） ==================
        "卫宁健康与乐普医疗2024年主营业务收入来源有何差异？",
        "上海悦心健康与卫宁健康在2024年医疗业务结构上有何不同？",

        # ================== AI / 具身智能（对比） ==================
        "奥比中光与中科曙光2024年AI相关业务收入来源有何不同？",
        "奥比中光和德赛西威在2024年智能感知技术应用上有何差异？",

        # ================== 汇总型（单公司多指标） ==================
        "总结中科曙光2024年的营业收入、净利润和研发投入情况。",
        "总结中国移动2024年的用户规模、收入和5G发展情况。",
        "总结赣锋锂业2024年的锂业务、储能业务和整体收入情况。",
        "总结中控技术2024年的收入结构和主要业务板块。",
        "总结东方财富2024年的主要收入来源和利润情况。",
        "总结德赛西威2024年的汽车电子业务布局和收入情况。",
        "总结卫宁健康2024年的医疗信息化业务发展情况。",
        "总结天能电池2024年的电池业务收入及市场情况。",
        "总结陕鼓动力2024年的核心业务板块及收入贡献。",
        "总结紫光国芯2024年的芯片业务收入及主要产品。",
        "总结山东天岳2024年的碳化硅业务发展情况。"
    ]
    
    mode = "auto"
    #mode = "manual"

    build_retrieval_gt_dataset(
        queries=queries,
        output_path="retrieval_gt_dataset.json",
        label_mode=mode,
        auto_score_threshold=0.65,
        fallback_first=True,
        search_mode="hybrid",
        use_reranker=True,
        retriever_score_threshold=0.3,
        device="cuda",
        preview_chars=500
    )

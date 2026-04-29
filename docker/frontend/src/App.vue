<template>
  <div class="app-container">
    <!-- 左侧：会话栏 -->
    <div class="left-panel">
      <h3>💬 会话列表</h3>
      <div class="session-item" @click="switchSession(s)" :class="{ active: s === currentSession }" v-for="s in sessions" :key="s">
        {{ s }}
      </div>
      <button class="clear-btn" @click="clearSession">清空当前会话</button>

      <h3 style="margin-top:20px;">📜 历史问题</h3>
      <div class="query-item" v-for="q in historyQueries" :key="q" @click="useQuery(q)">
        {{ q }}
      </div>
    </div>

    <!-- 中间：回答区 -->
    <div class="middle-panel">
      <div class="answer-container" ref="answerBox">
        <div v-for="(block, idx) in answerBlocks" :key="idx" class="answer-block">
          <div v-if="block.type === 'answer'" class="answer-content" v-html="highlightRefs(block.content)"></div>
          <div v-if="block.type === 'conclusion'" class="conclusion">{{ block.content }}</div>
          <div v-if="block.type === 'basis'" class="basis">{{ block.content }}</div>
          <div v-if="block.type === 'risk'" class="risk">{{ block.content }}</div>
        </div>
        <div class="loading" v-if="isLoading">思考中...</div>
      </div>

      <div class="input-bar">
        <input v-model="query" @keyup.enter="send" placeholder="输入问题..." />
        <button @click="send">发送</button>
      </div>
    </div>

    <!-- 右侧：证据区 -->
    <div class="right-panel">
      <h3>📄 证据片段</h3>
      <div class="evidence-item" v-for="(e, idx) in evidences" :key="idx">
        <div class="score">分数：{{ e.score.toFixed(2) }}</div>
        <div class="pdf">PDF：{{ e.pdf }}</div>
        <div class="page">页码：{{ e.page }}</div>
        <div class="content">{{ e.content }}</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'

const query = ref('')
const currentSession = ref('default')
const sessions = ref(['default'])
const historyQueries = ref([])
const answerBlocks = ref([])
const evidences = ref([])
const isLoading = ref(false)
const answerBox = ref(null)

// 发送问题
async function send() {
  if (!query.value) return
  historyQueries.value.push(query.value)
  answerBlocks.value = []
  evidences.value = []
  isLoading.value = true

  const resp = await fetch('http://localhost:8000/api/stream/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: query.value,
      session_id: currentSession.value
    })
  })

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    const text = decoder.decode(value)
    const lines = text.split('\n').filter(i => i.trim())

    for (let line of lines) {
      if (line.startsWith('data:')) {
        try {
          const obj = JSON.parse(line.slice(5))
          if (obj.type === 'answer') {
            answerBlocks.value.push({ type: 'answer', content: obj.content })
          }
          if (obj.type === 'evidences') {
            evidences.value = obj.list
          }
        } catch (e) {}
      }
    }
    await nextTick()
    answerBox.value.scrollTop = answerBox.value.scrollHeight
  }

  isLoading.value = false
  query.value = ''
}

// 引用高亮
function highlightRefs(content) {
  return content.replace(/\[(\d+)\]/g, '<span class="ref-highlight">[$1]</span>')
}

// 清空会话
async function clearSession() {
  await fetch('http://localhost:8000/api/clear_history', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: currentSession.value })
  })
  answerBlocks.value = []
  evidences.value = []
}

// 切换会话
function switchSession(s) {
  currentSession.value = s
}

// 快速使用历史问题
function useQuery(q) {
  query.value = q
}
</script>

<style scoped>
.app-container {
  display: flex;
  height: 100vh;
  width: 100vw;
  background: #f5f7fa;
}

.left-panel {
  width: 260px;
  background: #fff;
  border-right: 1px solid #eee;
  padding: 16px;
  overflow-y: auto;
}

.middle-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 20px;
}

.right-panel {
  width: 380px;
  background: #fff;
  border-left: 1px solid #eee;
  padding: 16px;
  overflow-y: auto;
}

.answer-container {
  flex: 1;
  background: white;
  border-radius: 8px;
  padding: 20px;
  overflow-y: auto;
  margin-bottom: 12px;
}

.input-bar {
  display: flex;
  gap: 8px;
}

input {
  flex: 1;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 8px;
}

button {
  padding: 12px 20px;
  background: #42b983;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
}

.session-item {
  padding: 10px 12px;
  border-radius: 6px;
  cursor: pointer;
  margin-bottom: 6px;
}

.session-item.active {
  background: #e6f7ee;
}

.evidence-item {
  padding: 12px;
  border-bottom: 1px solid #eee;
  margin-bottom: 10px;
}

.score {
  color: #42b983;
  font-size: 12px;
}

.pdf {
  font-size: 12px;
  color: #666;
}

.page {
  font-size: 12px;
  color: #666;
}

.content {
  margin-top: 6px;
  font-size: 13px;
  line-height: 1.5;
}

.ref-highlight {
  background: #fff3cd;
  padding: 2px 4px;
  border-radius: 4px;
  font-weight: bold;
}

.answer-content {
  line-height: 1.7;
  font-size: 15px;
}

.loading {
  color: #999;
  padding: 10px 0;
}
</style>
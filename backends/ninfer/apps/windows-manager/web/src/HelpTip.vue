<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, useId } from 'vue'
const props = defineProps<{ title: string; text: string; parameter?: string; language: 'zh' | 'en'; restart?: boolean }>()
const id = useId()
const trigger = ref<HTMLButtonElement>()
const tooltip = ref<HTMLElement>()
const open = ref(false)
const position = ref({ left: '12px', top: '12px' })
const accessibleName = computed(() => props.language === 'zh' ? `${props.title}：查看说明` : `Help for ${props.title}`)
let hideTimer: ReturnType<typeof setTimeout> | undefined
function cancelHide() { clearTimeout(hideTimer) }
async function show() {
  cancelHide(); open.value = true
  await nextTick()
  if (!trigger.value || !tooltip.value) return
  const anchor = trigger.value.getBoundingClientRect(), box = tooltip.value.getBoundingClientRect()
  const left = Math.max(12, Math.min(anchor.left - 12, innerWidth - box.width - 12))
  const below = anchor.bottom + 9
  const top = below + box.height <= innerHeight - 12 ? below : Math.max(12, anchor.top - box.height - 9)
  position.value = { left: `${left}px`, top: `${top}px` }
}
function hide() { cancelHide(); hideTimer = setTimeout(() => { open.value = false }, 150) }
function close() { cancelHide(); open.value = false }
onBeforeUnmount(cancelHide)
</script>
<template>
  <span class="help-anchor" @mouseenter="show" @mouseleave="hide">
    <button ref="trigger" type="button" class="help-button" :aria-label="accessibleName" :aria-describedby="open ? id : undefined" @focus="show" @blur="hide" @click.prevent.stop="show" @keydown.esc.prevent="close">?</button>
  </span>
  <Teleport to="body">
    <div v-if="open" :id="id" ref="tooltip" role="tooltip" class="help-tooltip" :style="position" @mouseenter="cancelHide" @mouseleave="hide">
      <strong>{{title}}</strong><code v-if="parameter">{{parameter}}</code>
      <p>{{text}}</p>
      <small v-if="restart">{{language === 'zh' ? '保存后在下次启动模型时生效；不会立即修改正在运行的服务。' : 'Saved changes apply the next time the model starts; the running service is not changed.'}}</small>
    </div>
  </Teleport>
</template>

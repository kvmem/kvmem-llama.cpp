<script setup lang="ts">
import { computed, ref } from 'vue'

type Series = { key: string; label: string; color: string; scale?: number }
type Point = { t: number; [key: string]: any }
const props = defineProps<{ title: string; unit: string; points: Point[]; series: Series[]; language: 'zh' | 'en'; ceiling?: number }>()
const hovered = ref<number | null>(null)
const W = 680, H = 200, left = 55, right = 15, top = 15, bottom = 33
const t = (zh: string, en: string) => props.language === 'zh' ? zh : en
const number = (value: number) => new Intl.NumberFormat(props.language === 'zh' ? 'zh-CN' : 'en-US', { maximumFractionDigits: Math.abs(value) < 10 ? 2 : 1 }).format(value)
const short = (value: number) => Math.abs(value) >= 1e6 ? number(value / 1e6) + 'M' : Math.abs(value) >= 1e3 ? number(value / 1e3) + 'K' : number(value)
const time = (value: number, full = false) => new Date(value).toLocaleTimeString(props.language === 'zh' ? 'zh-CN' : 'en-GB', { hour: '2-digit', minute: '2-digit', ...(full ? { second: '2-digit' as const } : {}) })
function value(point: Point | undefined, series: Series): number | null {
  const raw = point?.[series.key]
  return typeof raw === 'number' && Number.isFinite(raw) ? raw * (series.scale ?? 1) : null
}
const span = computed(() => {
  const start = props.points[0]?.t ?? Date.now(), end = props.points.at(-1)?.t ?? start
  return { start, end: end > start ? end : start + 2000 }
})
const maximum = computed(() => {
  let max = 0
  for (const point of props.points) for (const series of props.series) max = Math.max(max, value(point, series) ?? 0)
  if (props.ceiling && max <= props.ceiling) return props.ceiling
  if (!max) return 1
  const magnitude = 10 ** Math.floor(Math.log10(max))
  return Math.ceil(max / magnitude * 2) / 2 * magnitude
})
const hasData = computed(() => props.points.some(point => props.series.some(series => value(point, series) !== null)))
const x = (stamp: number) => left + (stamp - span.value.start) / (span.value.end - span.value.start) * (W - left - right)
const y = (v: number) => H - bottom - Math.max(0, v) / maximum.value * (H - top - bottom)

// Preserve each bucket's extrema and missing-data boundaries, instead of dropping narrow peaks.
function sampled(series: Series): { t: number; v: number | null }[] {
  const source = props.points
  const size = Math.max(1, Math.ceil(source.length / 180))
  const selected: { t: number; v: number | null }[] = []
  for (let start = 0; start < source.length; start += size) {
    const end = Math.min(source.length, start + size), indices = new Set([start, end - 1])
    let min = -1, max = -1, previous: number | null = null
    for (let i = start; i < end; i++) {
      const current = value(source[i], series)
      if (current !== null) {
        if (min < 0 || current < value(source[min], series)!) min = i
        if (max < 0 || current > value(source[max], series)!) max = i
      }
      if (i > start && (current === null) !== (previous === null)) { indices.add(i - 1); indices.add(i) }
      previous = current
    }
    if (min >= 0) indices.add(min)
    if (max >= 0) indices.add(max)
    for (const i of [...indices].sort((a, b) => a - b)) selected.push({ t: source[i].t, v: value(source[i], series) })
  }
  return selected
}
const paths = computed(() => props.series.map(series => {
  let drawing = false
  const parts: string[] = []
  for (const point of sampled(series)) {
    if (point.v === null) { drawing = false; continue }
    parts.push(`${drawing ? 'L' : 'M'}${x(point.t).toFixed(2)},${y(point.v).toFixed(2)}`)
    drawing = true
  }
  return { ...series, path: parts.join(' ') }
}))
const focus = computed(() => hovered.value === null ? undefined : props.points[Math.min(hovered.value, props.points.length - 1)])
function move(event: PointerEvent) {
  const rect = (event.currentTarget as SVGSVGElement).getBoundingClientRect()
  const target = span.value.start + Math.max(0, Math.min(1, ((event.clientX - rect.left) / rect.width * W - left) / (W - left - right))) * (span.value.end - span.value.start)
  let lo = 0, hi = props.points.length - 1
  while (lo < hi) { const middle = (lo + hi) >> 1; if (props.points[middle].t < target) lo = middle + 1; else hi = middle }
  if (lo > 0 && Math.abs(props.points[lo - 1].t - target) < Math.abs(props.points[lo]?.t - target)) lo--
  hovered.value = lo
}
function key(event: KeyboardEvent) {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
  event.preventDefault()
  const at = hovered.value ?? Math.max(0, props.points.length - 1)
  hovered.value = event.key === 'Home' ? 0 : event.key === 'End' ? props.points.length - 1 : Math.max(0, Math.min(props.points.length - 1, at + (event.key === 'ArrowLeft' ? -1 : 1)))
}
</script>

<template>
  <article class="trend-card">
    <div class="trend-heading"><h4>{{ title }}</h4><span>{{ unit }}</span></div>
    <div class="trend-surface">
      <svg :viewBox="`0 0 ${W} ${H}`" role="img" :aria-label="title + ' · ' + t('方向键查看采样', 'Use arrow keys to inspect samples')" tabindex="0" @pointermove="move" @pointerleave="hovered = null" @keydown="key" @blur="hovered = null">
        <g v-for="part in [0, .5, 1]" :key="part"><line :x1="left" :x2="W - right" :y1="y(maximum * part)" :y2="y(maximum * part)" stroke="#e9eee9"/><text :x="left - 9" :y="y(maximum * part) + 4" text-anchor="end">{{ short(maximum * part) }}</text></g>
        <path v-for="series in paths" :key="series.key" :d="series.path" :stroke="series.color" stroke-width="2" fill="none" vector-effect="non-scaling-stroke"/>
        <text :x="left" :y="H - 8">{{ props.points.length ? time(span.start) : '' }}</text><text :x="W - right" :y="H - 8" text-anchor="end">{{ props.points.length ? time(span.end) : '' }}</text>
        <line v-if="focus" :x1="x(focus.t)" :x2="x(focus.t)" :y1="top" :y2="H - bottom" stroke="#73978a" stroke-dasharray="3 3"/>
      </svg>
      <div v-if="!hasData" class="trend-empty">{{ t('等待有效采样', 'Waiting for valid samples') }}</div>
      <div v-if="focus" class="trend-tooltip" :style="{ left: `${Math.min(64, Math.max(3, x(focus.t) / W * 100))}%` }"><strong>{{ time(focus.t, true) }}</strong><div v-for="series in props.series" :key="series.key"><i :style="{ background: series.color }"></i>{{ series.label }} <b>{{ value(focus, series) === null ? '—' : number(value(focus, series)!) }} {{ unit }}</b></div></div>
    </div>
    <div class="trend-legend"><span v-for="series in props.series" :key="series.key"><i :style="{ background: series.color }"></i>{{ series.label }}</span></div>
  </article>
</template>

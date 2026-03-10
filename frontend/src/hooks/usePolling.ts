import { useEffect, useRef } from 'react'

export function usePolling(fn: () => void, intervalMs: number, immediate = true) {
  const savedFn = useRef(fn)
  savedFn.current = fn
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const startPolling = () => {
    if (document.hidden) return
    if (intervalRef.current) return
    intervalRef.current = setInterval(() => savedFn.current(), intervalMs)
  }

  const stopPolling = () => {
    if (!intervalRef.current) return
    clearInterval(intervalRef.current)
    intervalRef.current = null
  }

  useEffect(() => {
    const handler = () => {
      if (document.hidden) {
        stopPolling()
      } else {
        if (immediate) savedFn.current()
        startPolling()
      }
    }

    if (immediate && !document.hidden) {
      savedFn.current()
    }
    startPolling()
    document.addEventListener('visibilitychange', handler)

    return () => {
      stopPolling()
      document.removeEventListener('visibilitychange', handler)
    }
  }, [intervalMs, immediate])
}

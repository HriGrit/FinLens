import { useEffect, useRef } from 'react'

export function usePolling(fn: () => void, intervalMs: number, immediate = true) {
  const savedFn = useRef(fn)
  savedFn.current = fn

  useEffect(() => {
    if (immediate) savedFn.current()
    const id = setInterval(() => savedFn.current(), intervalMs)
    return () => clearInterval(id)
  }, [intervalMs, immediate])
}

import { render } from '@testing-library/react'
import { act } from 'react'
import { expect, test, vi } from 'vitest'

import { usePolling } from '../usePolling'

test('stops polling while tab is hidden and resumes on visibility change', () => {
  vi.useFakeTimers()

  let isHidden = false
  Object.defineProperty(document, 'hidden', {
    configurable: true,
    get: () => isHidden,
  })

  const callback = vi.fn()
  const interval = 1_000

  function PollingHarness() {
    usePolling(callback, interval, true)
    return null
  }

  render(<PollingHarness />)
  expect(callback).toHaveBeenCalledTimes(1)

  act(() => {
    vi.advanceTimersByTime(interval)
  })
  expect(callback).toHaveBeenCalledTimes(2)

  isHidden = true
  document.dispatchEvent(new Event('visibilitychange'))

  act(() => {
    vi.advanceTimersByTime(interval * 3)
  })
  expect(callback).toHaveBeenCalledTimes(2)

  isHidden = false
  document.dispatchEvent(new Event('visibilitychange'))

  act(() => {
    vi.advanceTimersByTime(10)
  })
  expect(callback).toHaveBeenCalledTimes(3)

  act(() => {
    vi.advanceTimersByTime(interval)
  })
  expect(callback).toHaveBeenCalledTimes(4)

  vi.useRealTimers()
})

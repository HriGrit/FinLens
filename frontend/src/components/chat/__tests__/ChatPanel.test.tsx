import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { act } from 'react'
import type { AxiosError } from 'axios'
import { beforeEach, expect, test, vi } from 'vitest'

import { ChatPanel } from '../ChatPanel'
import { useAppStore } from '../../../stores/useAppStore'
import { postChat } from '../../../api/client'
import type { ChatResponse } from '../../../stores/useAppStore'

vi.mock('../../../api/client', () => ({
  postChat: vi.fn(),
}))

const mockPostChat = vi.mocked(postChat)

beforeEach(() => {
  vi.clearAllMocks()
  useAppStore.setState({
    messages: [],
    isLoading: false,
    health: null,
    ingestion: null,
    freeModels: [],
    company: '',
    year: '',
    model: 'openrouter/stepfun/step-3.5-flash:free',
  })
})

test('shows backend error detail when chat returns a 404 response', async () => {
  const axiosError = new Error('Request failed with status code 404') as AxiosError<{ detail?: string }>
  axiosError.isAxiosError = true
  axiosError.code = 'ERR_BAD_REQUEST'
  axiosError.response = {
    status: 404,
    statusText: 'Not Found',
    headers: {},
    config: {} as never,
    data: { detail: 'No relevant context found for this query.' },
  }

  mockPostChat.mockRejectedValueOnce(axiosError)

  render(<ChatPanel />)

  const input = screen.getByPlaceholderText('Ask about financial filings...')
  const submit = screen.getByRole('button')
  fireEvent.change(input, { target: { value: 'query' } })
  fireEvent.click(submit)

  await waitFor(() => {
    expect(screen.getByText('Error: No relevant context found for this query.')).toBeInTheDocument()
  })
})

test('aborts in-flight chat request on unmount', async () => {
  let capturedSignal: AbortSignal | undefined
  mockPostChat.mockImplementation(async (_req: unknown, signal?: AbortSignal) => {
    capturedSignal = signal
    return new Promise<ChatResponse>(() => undefined)
  })

  const { unmount } = render(<ChatPanel />)

  const input = screen.getByPlaceholderText('Ask about financial filings...')
  const submit = screen.getByRole('button')
  fireEvent.change(input, { target: { value: 'long query that takes time' } })

  fireEvent.click(submit)
  unmount()

  await act(async () => {
    await Promise.resolve()
  })

  expect(capturedSignal?.aborted).toBe(true)
})

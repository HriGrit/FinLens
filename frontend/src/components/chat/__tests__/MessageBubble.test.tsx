import { render, screen } from '@testing-library/react'
import { expect, test } from 'vitest'

import { MessageBubble } from '../MessageBubble'
import { CitationCard } from '../CitationCard'
import type { Message } from '../../../stores/useAppStore'

test('renders assistant message content as markdown', () => {
  const message = {
    id: '1',
    role: 'assistant',
    content: '**bold** line with [OpenAI](https://openai.com) and list:\n- one\n- two',
  } satisfies Message

  render(<MessageBubble message={message} />)

  expect(screen.getByText('bold')).toBeInTheDocument()
  expect(screen.queryByText('**bold**')).not.toBeInTheDocument()
  expect(screen.getByText('one')).toBeInTheDocument()
  expect(screen.getByRole('link')).toHaveAttribute('href', 'https://openai.com')
})

const nullCitation = {
  index: 1,
  company: null,
  year: null,
  doc_type: null,
  page_number: null,
  filename: null,
}

test('renders null-safe citation values', () => {
  render(
    <CitationCard
      citation={nullCitation}
    />,
  )

  expect(screen.getByText(/Unknown company/)).toBeInTheDocument()
  expect(screen.getByText('Unknown document')).toBeInTheDocument()
  expect(screen.getByText(/Unknown year/)).toBeInTheDocument()
  expect(screen.getByText(/Unknown document type/)).toBeInTheDocument()
})

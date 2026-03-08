import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#0d0f12',
        surface: '#141720',
        border: '#1e2330',
        amber: {
          400: '#fbbf24',
          500: '#f59e0b',
        },
        emerald: {
          400: '#34d399',
          500: '#10b981',
        },
        red: {
          400: '#f87171',
          500: '#ef4444',
        },
        muted: '#6b7280',
        'text-primary': '#e2e8f0',
        'text-secondary': '#94a3b8',
      },
      fontFamily: {
        mono: ['"IBM Plex Mono"', 'monospace'],
        sans: ['Inter', 'sans-serif'],
      },
      backgroundImage: {
        'grid-pattern':
          'linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)',
      },
      backgroundSize: {
        grid: '24px 24px',
      },
    },
  },
  plugins: [],
}

export default config

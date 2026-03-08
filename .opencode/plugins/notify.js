export const NotifyPlugin = async ({ $ }) => {
  const notify = async (message, title = "OpenCode", sound = "Glass") => {
    const safe = message.replace(/'/g, "\\'")
    await $`osascript -e ${"display notification \"" + safe + "\" with title \"" + title + "\" sound name \"" + sound + "\""}`
  }

  return {
    event: async ({ event }) => {
      // Agent finished / waiting for input
      if (event.type === "session.idle") {
        await notify("Agent is done or waiting for you", "OpenCode", "Glass")
      }

      // Agent needs explicit permission (file write, bash, etc.)
      if (event.type === "permission.asked") {
        await notify("Agent needs your approval", "OpenCode \u2757", "Ping")
      }

      // Session hit an error
      if (event.type === "session.error") {
        await notify("Session error occurred", "OpenCode \u26a0\ufe0f", "Basso")
      }
    },
  }
}

I'm thinking we should add a general harness-driven tool for compacting overlay large files.

- devstral 2 as model
- prompt focused on reducing redundancy, keeping tone of voice and significant points
- Then run this on written/appended files that are overly long (especially thinking.md)

This would be an automatic step that is allowed to interrupt/pause instance processing, even within a retry etc. it should be part of tool invocation handling, most likely, applied to the response from llm calls if that response if very long (or existing + appended is over the limit).

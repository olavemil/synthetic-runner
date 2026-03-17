# Simplify and tighten schedule restrictions

To ensure all instances get runtime, we should probably enforce both a minimum and maximum for thinking and messaging per heartbeat. Perhaps we could clean this up across the entire setup?

- external triggers is system scheduler/cron at 5 minute intervals
- separate schedules
  - instances "guaranteed" to get runtime for thinking at some heartbeat (assuming computer is on)
    - increments should probably be 3 times what they are today
  - instances allowed runtime for N thinking sessions at T interval after messages received/responded to
- constraints
  - M replies allowed between scheduled heartbeats
  - up to N thinking sessions after messages

Intended result:

- no messages -> infrequent but mostly regular thinking sessions (a few times per day, per instance)
- one message received -> allowed to respond, but have to wait for thinking to reply again, and only if new messages are received at that time

The system would therefore not be starved for resources when "nothing" is happening, and the instances would be less chatty to avoid crowding out other voices with repeated opinions.
We should also limit the on_message thinking to either exclude or reduce likelihood of creative and organization phases. They should be allowed to read the knowledge and creations, but not have focus on managing it.

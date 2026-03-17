# Scheduling Rework Implementation Plan

## Current State Analysis

### Current System Flow
1. **OS Scheduler Level** (`scheduling.py`):
   - Generates launchd/systemd/crontab files
   - Fixed intervals: `check_interval=300s` (5 min), `work_interval=60s` (1 min)
   - No per-instance constraints

2. **Unified Scheduler** (`scheduler.py`):
   - Single process polls messaging adapters (`_poll_reactive`)
   - Checks cron schedules for entry points (`_check_schedules`)
   - Dispatches handlers with per-instance locking
   - Entry points have: `schedule` (cron) and `trigger` (reactive)

3. **Instance Configuration** (`config.py`):
   - `InstanceConfig.schedule`: dict of cron overrides per entry point
   - No budget/constraint tracking

### Current Entry Point Model
```
EntryPoint(
  name: str,           # "on_message", "heartbeat"
  handler: Callable,
  schedule: str | None = None,  # cron expression
  trigger: str | None = None     # unused currently
)
```

---

## Proposed Changes (from scheduling_rework.md)

### Goals
1. **Resource fairness**: Ensure all instances get runtime, prevent starvation
2. **Reduced chattiness**: Limit reply budgets to avoid crowding out other voices
3. **Guaranteed thinking**: Regular thinking even when silent (3x more frequent than today)
4. **Reactive budgets**: Limited thinking sessions after messages received

### Core Constraints
- **Guaranteed thinking**: Periodic heartbeats with thinking runtime (e.g., every 4-8 hours per instance)
- **Reply budget**: M replies allowed between guaranteed thinking windows
- **Reactive thinking budget**: Up to N thinking sessions after messages, at interval T
- **Phase gating**: on_message thinking should reduce/exclude creative & organization phases

### Intended Behavior
- No messages → infrequent but regular thinking (e.g., 3x/day)
- Message received → allowed to respond, must wait for next thinking window
- Result: balanced resource usage, fewer redundant messages

---

## Implementation Plan

### Phase 1: Config & Storage Layer

#### 1.1 Extend `InstanceConfig` with scheduling constraints
**File**: `library/harness/config.py`

Add new dataclass:
```python
@dataclass
class SchedulingConstraints:
    # Guaranteed thinking windows
    guaranteed_thinking_interval: int = 14400  # seconds (4 hours)

    # Reply budget between guaranteed thinking windows
    max_replies_per_window: int = 3

    # Reactive thinking after messages
    reactive_thinking_max_sessions: int = 2  # max thinking sessions per message event
    reactive_thinking_cooldown: int = 900    # seconds (15 min) between reactive sessions

    # Phase restrictions for on_message
    on_message_thinking_phases: set[str] = None  # e.g., {"THINKING"}, exclude COMPOSING/REVIEWING
```

Add to `InstanceConfig`:
```python
@dataclass
class InstanceConfig:
    # ... existing fields ...
    scheduling_constraints: SchedulingConstraints | None = None
```

#### 1.2 Create scheduler state tracking in Store
**File**: `library/harness/scheduler.py` → new method `_init_scheduler_state()`

Track per-instance in `NamespacedStore` (key: `scheduler:{instance_id}:{constraint}`):
- `replies_this_window`: int (reset on guaranteed thinking)
- `reactive_sessions_since_message`: int (reset on message)
- `last_reactive_session_time`: float (timestamp, for cooldown)
- `last_guaranteed_thinking_time`: float (timestamp, to determine next window)
- `message_event_count`: int (track events received in current window)

---

### Phase 2: Scheduler Logic

#### 2.1 Update `_check_schedules()` in scheduler.py
Add budget enforcement:

```python
def _check_schedules(self) -> None:
    """Check cron schedules, respecting per-instance budgets."""
    now = time.time()

    for instance_config in self._registry.list_instances():
        instance_id = instance_config.instance_id
        constraints = instance_config.scheduling_constraints or self._default_constraints()

        # Track reply count in this window
        store = NamespacedStore(self._store_db, f"scheduler:{instance_id}")
        replies_this_window = store.get("replies_this_window") or 0

        # For on_message: check reply budget before dispatching
        # For heartbeat: reset counters when it fires
```

#### 2.2 Add `_check_reply_budget()` helper
```python
def _check_reply_budget(self, instance_id: str) -> bool:
    """Check if instance can send another reply."""
    store = NamespacedStore(self._store_db, f"scheduler:{instance_id}")
    constraints = self._get_constraints(instance_id)
    replies = store.get("replies_this_window") or 0
    return replies < constraints.max_replies_per_window
```

#### 2.3 Add `_check_reactive_thinking_budget()` helper
```python
def _check_reactive_thinking_budget(self, instance_id: str, now: float) -> bool:
    """Check if instance can run reactive thinking session."""
    store = NamespacedStore(self._store_db, f"scheduler:{instance_id}")
    constraints = self._get_constraints(instance_id)

    sessions = store.get("reactive_sessions_since_message") or 0
    last_session_time = store.get("last_reactive_session_time") or 0

    if sessions >= constraints.reactive_thinking_max_sessions:
        return False
    if (now - last_session_time) < constraints.reactive_thinking_cooldown:
        return False
    return True
```

#### 2.4 Update `_poll_reactive()` to enforce budgets
**File**: `library/harness/scheduler.py`

When dispatching `on_message`:
- Check reply budget via `_check_reply_budget()` → increment counter on dispatch
- Check phase restrictions → pass phase parameter to handler
- Track message event count

```python
def _poll_reactive(self) -> None:
    """Poll for new events, enforce budgets before dispatching."""
    # ... existing polling code ...

    if normalized_events and token is not None:
        entity_id = instance_config.messaging.entity_id
        own_events = [e for e in normalized_events if e.sender != entity_id]
        if own_events:
            # Enforce reply budget
            if self._check_reply_budget(instance_id):
                self._dispatch(instance_id, "on_message",
                             events=own_events,
                             on_message_phase="THINKING")  # restricted phase
                store = NamespacedStore(self._store_db, f"scheduler:{instance_id}")
                store.put("replies_this_window",
                         (store.get("replies_this_window") or 0) + 1)
```

#### 2.5 Add guaranteed thinking window reset
In `_check_schedules()`, when `heartbeat` entry point fires:

```python
# After dispatching heartbeat, reset budgets
store = NamespacedStore(self._store_db, f"scheduler:{instance_id}")
store.put("replies_this_window", 0)
store.put("reactive_sessions_since_message", 0)
store.put("last_guaranteed_thinking_time", now)
```

---

### Phase 3: Entry Point & Handler Updates

#### 3.1 Add phase parameter to on_message signature
**Files**: All species' `on_message()` handlers

Current:
```python
def on_message(ctx: InstanceContext, events: list[Event]) -> None:
```

Updated:
```python
def on_message(ctx: InstanceContext, events: list[Event], on_message_phase: str | None = None) -> None:
```

#### 3.2 Use phase filtering in on_message
**File**: `library/tools/phases.py` already has `get_tools_for_phase()`

In each species' `on_message()`, when calling thinking/organizing:
```python
# If on_message_phase is set, restrict tools
if on_message_phase:
    tools = get_tools_for_phase(on_message_phase)
else:
    tools = get_tools_for_phase("THINKING")
```

#### 3.3 Update heartbeat schedule
**File**: Species manifests (e.g., `library/species/consilium/__init__.py`)

Current: `schedule="0 * * * *"` (every hour)

Proposed: Make configurable via instance config or adjust base to ~4-8 hours:
```python
EntryPoint(name="heartbeat", handler=heartbeat, schedule="0 */4 * * *"),  # every 4 hours
```

Or better: let instance config override via `schedule: { heartbeat: "0 */4 * * *" }`

---

### Phase 4: Instance Configuration

#### 4.1 Update instance YAML format
Example `config/instances/mybot.yaml`:

```yaml
instance_id: mybot
species: consilium
provider: anthropic
model: claude-3-5-sonnet-20241022
messaging:
  adapter: matrix
  entity_id: "@mybot:example.com"
  access_token: ${MYBOT_TOKEN}
  spaces:
    - name: main
      handle: !room:example.com

schedule:
  heartbeat: "0 */4 * * *"  # every 4 hours (guaranteed thinking)

scheduling_constraints:
  guaranteed_thinking_interval: 14400  # 4 hours
  max_replies_per_window: 3
  reactive_thinking_max_sessions: 2
  reactive_thinking_cooldown: 900  # 15 min between sessions
  on_message_thinking_phases: ["THINKING"]  # exclude COMPOSING, REVIEWING
```

#### 4.2 Add defaults to HarnessConfig
**File**: `library/harness/config.py`

```python
def _default_scheduling_constraints() -> SchedulingConstraints:
    return SchedulingConstraints(
        guaranteed_thinking_interval=14400,  # 4 hours
        max_replies_per_window=3,
        reactive_thinking_max_sessions=2,
        reactive_thinking_cooldown=900,
        on_message_thinking_phases={"THINKING"},
    )
```

---

### Phase 5: Testing

#### 5.1 Unit tests for constraint checking
**File**: `tests/test_scheduler_constraints.py` (new)

- Test `_check_reply_budget()` enforcement
- Test `_check_reactive_thinking_budget()` cooldown
- Test budget reset on heartbeat
- Test phase filtering on on_message

#### 5.2 Integration tests
**File**: `tests/test_scheduler_integration.py` (new or extend)

- Simulate rapid messages → enforce reply budget
- Simulate heartbeat → reset budgets
- Verify on_message tools are gated by phase

---

## Summary of Changes

| File | Changes |
|------|---------|
| `library/harness/config.py` | Add `SchedulingConstraints` dataclass, extend `InstanceConfig` |
| `library/harness/scheduler.py` | Add budget tracking, constraint checking in `_poll_reactive()` / `_check_schedules()` |
| `library/species/*/\_\_init\_\_.py` | Update `on_message()` signatures to accept `on_message_phase` param |
| Species handlers (all 6) | Implement phase filtering when `on_message_phase` is set |
| Instance YAML files | Add `scheduling_constraints` section |
| `tests/test_scheduler_*.py` | New tests for constraint enforcement |

---

## Open Questions / Design Decisions

1. **Default intervals**: Should guaranteed thinking be per-instance or species-wide?
   - Proposal: Per-instance in `scheduling_constraints` with sensible defaults

2. **On-message phase restriction**: Disable completely or just lower priority?
   - Proposal: Pass phase param; species decide how to apply (could use `get_tools_for_phase`)

3. **Reaction to new messages during window**: Should a new message reset reactive counters?
   - Proposal: Track `last_message_time`; new messages don't reset but allow new budget

4. **Store cleanup**: When/how to prune old scheduler state?
   - Proposal: Lazy cleanup on heartbeat; old entries auto-expire after 48h

5. **Backward compatibility**: Instances without `scheduling_constraints`?
   - Proposal: Use defaults; non-breaking change

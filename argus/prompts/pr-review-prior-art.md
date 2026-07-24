## Prior Art and Simplification Check

For each piece of new functionality in the diff, check whether it reinvents something that already exists. Be naive on purpose -- do not assume the author knew about the alternative.

### 1. Already in rh-lib?
Grep rh_lib/integrations/, rh_lib/utils/, rh_lib/database/, rh_lib/config/. If the PR adds a client, helper, or wrapper for something rh-lib handles, flag it.

### 2. Framework handles this natively?
Use Context7 to verify before concluding code is novel:
- Prefect: @task retries/parallelism, CronSchedule, artifacts
- LangGraph: Send() fan-out, RetryPolicy, AsyncPostgresSaver checkpointer, Command routing
- FastAPI: Depends(), middleware, BackgroundTasks
- SQLAlchemy async: selectinload/joinedload for N+1, async session patterns
- Slack Bolt SDK: Event subscriptions, slash commands, interactive components

### 3. Better library already installed?
Check redesign-health-data/pyproject.toml first. Common: custom retry -> tenacity, rate limiting -> ratelimit, date parsing -> python-dateutil, validation -> pydantic.

### 4. Should be extracted to rh-lib?
New code that interacts with an external service and could serve multiple projects belongs in rh_lib/integrations/ or rh_lib/utils/.

### Rules
- Only flag with a concrete alternative (name the file, class, API, or library)
- Do not manufacture concerns -- if the code is appropriate, do not flag it
- Context7 lookup is mandatory for framework capability claims
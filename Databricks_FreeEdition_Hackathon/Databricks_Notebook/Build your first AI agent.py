# Databricks notebook source
# MAGIC %md
# MAGIC # Quickstart: Build, test, and deploy an agent using Mosaic AI Agent Framework
# MAGIC This quickstart notebook demonstrates how to build, test, and deploy a generative AI agent ([AWS](https://docs.databricks.com/aws/en/generative-ai/guide/introduction-generative-ai-apps#what-are-gen-ai-apps) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/guide/introduction-generative-ai-apps#what-are-gen-ai-apps) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/guide/introduction-generative-ai-apps)) using Mosaic AI Agent Framework ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/build-genai-apps#-mosaic-ai-agent-framework) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/build-genai-apps#-mosaic-ai-agent-framework) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/build-genai-apps#-mosaic-ai-agent-framework)) on Databricks

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Define and test an agent
# MAGIC This section defines and tests a simple agent with the following attributes:
# MAGIC
# MAGIC - The agent uses an LLM served on Databricks Foundation Model API ([AWS](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/) | [GCP](https://docs.databricks.com/gcp/en/machine-learning/foundation-model-apis))
# MAGIC - The agent has access to a single tool, the built-in Python code interpreter tool on Databricks Unity Catalog. It can use this tool to run LLM-generated code in order to respond to user questions. ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/code-interpreter-tools#built-in-python-executor-tool) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/code-interpreter-tools) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/code-interpreter-tools))
# MAGIC
# MAGIC We will use `databricks_openai` SDK ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent#requirements) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/author-agent#requirements) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/author-agent#requirements)) to query the LLM endpoint.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow databricks-openai databricks-agents

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# The snippet below tries to pick the first LLM API available in your Databricks workspace
# from a set of candidates. You can override and simplify it
# to just specify LLM_ENDPOINT_NAME.
LLM_ENDPOINT_NAME = None

from databricks.sdk import WorkspaceClient
def is_endpoint_available(endpoint_name):
  try:
    client = WorkspaceClient().serving_endpoints.get_open_ai_client()
    client.chat.completions.create(model=endpoint_name, messages=[{"role": "user", "content": "What is AI?"}])
    return True
  except Exception:
    return False
  
client = WorkspaceClient()
for candidate_endpoint_name in ["databricks-claude-3-7-sonnet", "databricks-meta-llama-3-3-70b-instruct"]:
    if is_endpoint_available(candidate_endpoint_name):
      LLM_ENDPOINT_NAME = candidate_endpoint_name
assert LLM_ENDPOINT_NAME is not None, "Please specify LLM_ENDPOINT_NAME"

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Replace these with your actual catalog and schema names
catalog_name = "workspace"
schema_name = "default"

functions = w.functions.list(catalog_name, schema_name)
for fn in functions:
    print(fn.full_name)


# COMMAND ----------

# MAGIC %md
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC
# MAGIC w = WorkspaceClient()
# MAGIC
# MAGIC print("Catalogs:")
# MAGIC for c in w.catalogs.list():
# MAGIC     print(" -", c.name)
# MAGIC
# MAGIC print("\nSchemas:")
# MAGIC for s in w.schemas.list("system"):  # you can change "system" to another catalog name
# MAGIC     print(" -", s.name)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC
# MAGIC w = WorkspaceClient()
# MAGIC
# MAGIC print("Writable catalogs:")
# MAGIC for c in w.catalogs.list():
# MAGIC     print("-", c.name)
# MAGIC
# MAGIC print("\nSchemas in 'default':")
# MAGIC for s in w.schemas.list("workspace"):
# MAGIC     print("-", s.name)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.default;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC CREATE OR REPLACE FUNCTION workspace.default.python_exec(code STRING)
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON AS $$
# MAGIC exec_locals = {}
# MAGIC exec(code, {}, exec_locals)
# MAGIC return str(exec_locals)
# MAGIC $$;

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION workspace.default.python_exec(code STRING)
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON AS $$
# MAGIC import io, contextlib, json
# MAGIC
# MAGIC def safe_serialize(obj):
# MAGIC     try:
# MAGIC         return obj if isinstance(obj, (str, int, float, bool, list, dict, type(None))) else str(obj)
# MAGIC     except Exception:
# MAGIC         return str(obj)
# MAGIC
# MAGIC buffer = io.StringIO()
# MAGIC exec_locals = {}
# MAGIC
# MAGIC with contextlib.redirect_stdout(buffer):
# MAGIC     exec(code, {}, exec_locals)
# MAGIC
# MAGIC # Grab result if defined
# MAGIC if "result" in exec_locals:
# MAGIC     return json.dumps({"result": safe_serialize(exec_locals["result"])})
# MAGIC
# MAGIC # fallback: return stdout if no result
# MAGIC output = buffer.getvalue().strip()
# MAGIC return json.dumps({"result": output})
# MAGIC $$;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC CREATE OR REPLACE FUNCTION workspace.default.python_exec(code STRING)
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON AS $$
# MAGIC import io, contextlib, json
# MAGIC
# MAGIC buffer = io.StringIO()
# MAGIC exec_locals = {}
# MAGIC
# MAGIC with contextlib.redirect_stdout(buffer):
# MAGIC     exec(code, {}, exec_locals)
# MAGIC
# MAGIC output = buffer.getvalue().strip()
# MAGIC
# MAGIC # Only return 'result' if it exists
# MAGIC if "result" in exec_locals:
# MAGIC     try:
# MAGIC         return json.dumps({"result": exec_locals["result"], "stdout": output})
# MAGIC     except TypeError:
# MAGIC         # If 'result' is not JSON-serializable, return its string representation
# MAGIC         return json.dumps({"result": str(exec_locals["result"]), "stdout": output})
# MAGIC
# MAGIC # If no result, return only stdout
# MAGIC return json.dumps({"stdout": output})
# MAGIC $$;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC CREATE OR REPLACE FUNCTION workspace.default.python_exec(code STRING)
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON AS $$
# MAGIC import io, contextlib, json
# MAGIC
# MAGIC buffer = io.StringIO()
# MAGIC exec_locals = {}
# MAGIC
# MAGIC with contextlib.redirect_stdout(buffer):
# MAGIC     exec(code, {}, exec_locals)
# MAGIC
# MAGIC output = buffer.getvalue().strip()
# MAGIC # Prefer a variable named "result" if defined
# MAGIC if "result" in exec_locals:
# MAGIC     return json.dumps({"result": exec_locals["result"], "stdout": output})
# MAGIC # If no "result", return all locals
# MAGIC elif len(exec_locals) > 0:
# MAGIC     return json.dumps({"locals": exec_locals, "stdout": output})
# MAGIC else:
# MAGIC     return json.dumps({"stdout": output})
# MAGIC $$;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC CREATE FUNCTION main.default.python_exec(code STRING)
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON AS $$
# MAGIC exec_locals = {}
# MAGIC exec(code, {}, exec_locals)
# MAGIC return str(exec_locals)
# MAGIC $$;

# COMMAND ----------



# COMMAND ----------

import json
import math
import mlflow
from databricks.sdk import WorkspaceClient
from databricks_openai import UCFunctionToolkit, DatabricksFunctionClient

# Automatically log traces from LLM calls for ease of debugging
mlflow.openai.autolog()

# Get an OpenAI client configured to talk to Databricks model serving endpoints
# We'll use this to query an LLM in our agent
openai_client = WorkspaceClient().serving_endpoints.get_open_ai_client()

# Load Databricks built-in tools (a stateless Python code interpreter tool)
client = DatabricksFunctionClient()
builtin_tools = UCFunctionToolkit(
    function_names=["workspace.default.python_exec"], client=client
).tools
#builtin_tools = UCFunctionToolkit(
#    function_names=["system.ai.python_exec"], client=client
#).tools
for tool in builtin_tools:
    del tool["function"]["strict"]

def call_tool(tool_name, parameters):
    if tool_name == "workspace__default__python_exec":
        return DatabricksFunctionClient().execute_function(
            "workspace.default.python_exec", parameters=parameters
        )
    raise ValueError(f"Unknown tool: {tool_name}")

 


# COMMAND ----------

import json

def run_agent(prompt):
    """
    Sends a prompt to the LLM, executes any tool if called,
    and returns a list of messages with a proper assistant reply.
    """
    result_msgs = []

    response = openai_client.chat.completions.create(
        model=LLM_ENDPOINT_NAME,
        messages=[{"role": "user", "content": prompt}],
        tools=builtin_tools,
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        # LLM wants to call a tool
        call = msg.tool_calls[0]
        tool_result = call_tool(call.function.name, json.loads(call.function.arguments))

        # Parse the JSON result
        try:
            parsed = json.loads(tool_result.value)
            content = parsed.get("result") or parsed.get("stdout") or ""
        except Exception:
            content = tool_result.value

        # Only append a single assistant message with the tool output
        result_msgs.append({
            "role": "assistant",
            "content": str(content),
            "name": call.function.name,
            "tool_call_id": call.id
        })
    else:
        # No tool call: use the LLM's message as-is
        result_msgs.append(msg.to_dict())

    return result_msgs


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC import json
# MAGIC import mlflow
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks_openai import UCFunctionToolkit, DatabricksFunctionClient
# MAGIC
# MAGIC # Automatically log traces from LLM calls for ease of debugging
# MAGIC mlflow.openai.autolog()
# MAGIC
# MAGIC # Get an OpenAI client configured to talk to Databricks model serving endpoints
# MAGIC # We'll use this to query an LLM in our agent
# MAGIC openai_client = WorkspaceClient().serving_endpoints.get_open_ai_client()
# MAGIC
# MAGIC # Load Databricks built-in tools (a stateless Python code interpreter tool)
# MAGIC client = DatabricksFunctionClient()
# MAGIC builtin_tools = UCFunctionToolkit(
# MAGIC     function_names=["workspace.default.python_exec"], client=client
# MAGIC ).tools
# MAGIC #builtin_tools = UCFunctionToolkit(
# MAGIC #    function_names=["system.ai.python_exec"], client=client
# MAGIC #).tools
# MAGIC for tool in builtin_tools:
# MAGIC     del tool["function"]["strict"]
# MAGIC
# MAGIC def call_tool(tool_name, parameters):
# MAGIC     if tool_name == "workspace__default__python_exec":
# MAGIC         return DatabricksFunctionClient().execute_function(
# MAGIC             "workspace.default.python_exec", parameters=parameters
# MAGIC         )
# MAGIC     raise ValueError(f"Unknown tool: {tool_name}")
# MAGIC
# MAGIC def call_toolOld(tool_name, parameters):
# MAGIC     if tool_name == "system__ai__python_exec":
# MAGIC         return DatabricksFunctionClient().execute_function(
# MAGIC             "system.ai.python_exec", parameters=parameters
# MAGIC         )
# MAGIC     raise ValueError(f"Unknown tool: {tool_name}")
# MAGIC
# MAGIC
# MAGIC def run_agent(prompt):
# MAGIC     """
# MAGIC     Send a user prompt to the LLM, and return a list of LLM response messages
# MAGIC     The LLM is allowed to call the code interpreter tool if needed, to respond to the user
# MAGIC     """
# MAGIC     result_msgs = []
# MAGIC     response = openai_client.chat.completions.create(
# MAGIC         model=LLM_ENDPOINT_NAME,
# MAGIC         messages=[{"role": "user", "content": prompt}],
# MAGIC         tools=builtin_tools,
# MAGIC     )
# MAGIC     msg = response.choices[0].message
# MAGIC     result_msgs.append(msg.to_dict())
# MAGIC
# MAGIC     # If the model executed a tool, call it
# MAGIC     if msg.tool_calls:
# MAGIC         call = msg.tool_calls[0]
# MAGIC         tool_result = call_tool(call.function.name, json.loads(call.function.arguments))
# MAGIC         result_msgs.append(
# MAGIC             {
# MAGIC                 "role": "tool",
# MAGIC                 "content": tool_result.value,
# MAGIC                 "name": call.function.name,
# MAGIC                 "tool_call_id": call.id,
# MAGIC             }
# MAGIC         )
# MAGIC     return result_msgs

# COMMAND ----------

# MAGIC %md
# MAGIC from databricks_openai import DatabricksFunctionClient
# MAGIC client = DatabricksFunctionClient()
# MAGIC
# MAGIC result = client.execute_function(
# MAGIC     "workspace.default.python_exec",
# MAGIC     parameters={"code": "x = 2 + 3\nprint(x)"}
# MAGIC )
# MAGIC print(result.value)

# COMMAND ----------

import math
answer = run_agent("What are the colors of the rainbow?")
for message in answer:
    print(f'{message["role"]}: {message["content"]}')

# COMMAND ----------

import math
answer = run_agent("What is the square root of 429?")
for message in answer:
    print(f'{message["role"]}: {message["content"]}')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Prepare agent code for logging
# MAGIC
# MAGIC Wrap your agent definition in MLflow’s [ChatAgent interface](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatAgent) to prepare your code for logging.
# MAGIC
# MAGIC By using MLflow’s standard agent authoring interface, you get built-in UIs for chatting with your agent and sharing it with others after deployment. ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent#-use-chatagent-to-author-agents) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/author-agent) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/author-agent))

# COMMAND ----------

import uuid
import mlflow
from typing import Any, Optional

from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatContext

mlflow.openai.autolog()

class QuickstartAgent(ChatAgent):
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        # 1. Extract the last user prompt from the input messages
        prompt = messages[-1].content

        # 2. Call run_agent to get back a list of response messages
        raw_msgs = run_agent(prompt)

        # 3. Map each response message into a ChatAgentMessage and return
        # the response
        out = []
        for m in raw_msgs:
            out.append(ChatAgentMessage(id=uuid.uuid4().hex, **m))

        return ChatAgentResponse(messages=out)

# COMMAND ----------

AGENT = QuickstartAgent()
for response_message in AGENT.predict(
    {"messages": [{"role": "user", "content": "What's the square root of 429?"}]}
).messages:
    print(f"role: {response_message.role}, content: {response_message.content}")

# COMMAND ----------

# MAGIC %md ## Log the agent
# MAGIC
# MAGIC Log the agent and register it to Unity Catalog as a model ([AWS](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/manage-model-lifecycle/) | [GCP](https://docs.databricks.com/gcp/en/machine-learning/manage-model-lifecycle/)). This step packages the agent code and its dependencies into a single artifact to deploy it to a serving endpoint.
# MAGIC
# MAGIC The following code cells do the following:
# MAGIC
# MAGIC 1. Copy the agent code from above and combine it into a single cell.
# MAGIC 1. Add the `%%writefile` cell magic command at the top of the cell to save the agent code to a file called `quickstart_agent.py`.
# MAGIC 1. Add a [mlflow.models.set_model()](https://mlflow.org/docs/latest/model#models-from-code) call to the bottom of the cell. This tells MLflow which Python agent object to use for making predictions when your agent is deployed.
# MAGIC 1. Log the agent code in the `quickstart_agent.py` file using MLflow APIs ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/log-agent) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/log-agent) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/log-agent)).

# COMMAND ----------

# MAGIC %%writefile quickstart_agent.py
# MAGIC import json
# MAGIC import uuid
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks_openai import UCFunctionToolkit, DatabricksFunctionClient
# MAGIC from typing import Any, Optional
# MAGIC import mlflow
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatContext
# MAGIC
# MAGIC # Configure OpenAI client for Databricks model serving
# MAGIC openai_client = WorkspaceClient().serving_endpoints.get_open_ai_client()
# MAGIC
# MAGIC # ✅ Use your confirmed, existing LLM endpoint
# MAGIC LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC
# MAGIC # Enable automatic tracing of LLM calls
# MAGIC mlflow.openai.autolog()
# MAGIC
# MAGIC # Load Databricks built-in tools (Python code interpreter)
# MAGIC client = DatabricksFunctionClient()
# MAGIC builtin_tools = UCFunctionToolkit(
# MAGIC     function_names=["workspace.default.python_exec"], client=client
# MAGIC ).tools
# MAGIC for tool in builtin_tools:
# MAGIC     del tool["function"]["strict"]
# MAGIC
# MAGIC def call_tool(tool_name, parameters):
# MAGIC     if tool_name == "workspace__default__python_exec":
# MAGIC         return DatabricksFunctionClient().execute_function(
# MAGIC             "workspace.default.python_exec", parameters=parameters
# MAGIC         )
# MAGIC     raise ValueError(f"Unknown tool: {tool_name}")
# MAGIC
# MAGIC def run_agent(prompt):
# MAGIC     """Send user prompt to the LLM and handle tool execution if requested."""
# MAGIC     result_msgs = []
# MAGIC     response = openai_client.chat.completions.create(
# MAGIC         model=LLM_ENDPOINT_NAME,
# MAGIC         messages=[{"role": "user", "content": prompt}],
# MAGIC         tools=builtin_tools,
# MAGIC     )
# MAGIC     msg = response.choices[0].message
# MAGIC     result_msgs.append(msg.to_dict())
# MAGIC
# MAGIC     if msg.tool_calls:
# MAGIC         call = msg.tool_calls[0]
# MAGIC         tool_result = call_tool(
# MAGIC             call.function.name, json.loads(call.function.arguments)
# MAGIC         )
# MAGIC         result_msgs.append({
# MAGIC             "role": "tool",
# MAGIC             "content": tool_result.value,
# MAGIC             "name": call.function.name,
# MAGIC             "tool_call_id": call.id
# MAGIC         })
# MAGIC     return result_msgs
# MAGIC
# MAGIC class QuickstartAgent(ChatAgent):
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         prompt = messages[-1].content
# MAGIC         raw_msgs = run_agent(prompt)
# MAGIC         out = [
# MAGIC             ChatAgentMessage(id=uuid.uuid4().hex, **m)
# MAGIC             for m in raw_msgs
# MAGIC         ]
# MAGIC         return ChatAgentResponse(messages=out)
# MAGIC
# MAGIC AGENT = QuickstartAgent()
# MAGIC

# COMMAND ----------

# MAGIC %%writefile quickstart_agent.py
# MAGIC import json
# MAGIC import uuid
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks_openai import UCFunctionToolkit, DatabricksFunctionClient
# MAGIC from typing import Any, Optional
# MAGIC
# MAGIC import mlflow
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatContext
# MAGIC
# MAGIC # Set LLM endpoint explicitly or pick first available
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC
# MAGIC # Enable automatic tracing of LLM calls
# MAGIC mlflow.openai.autolog()
# MAGIC
# MAGIC # OpenAI client for agent
# MAGIC openai_client = WorkspaceClient().serving_endpoints.get_open_ai_client()
# MAGIC
# MAGIC # Load Databricks built-in tools (stateless Python code interpreter)
# MAGIC client = DatabricksFunctionClient()
# MAGIC builtin_tools = UCFunctionToolkit(function_names=["workspace.default.python_exec"], client=client).tools
# MAGIC for tool in builtin_tools:
# MAGIC     del tool["function"]["strict"]
# MAGIC
# MAGIC def call_tool(tool_name, parameters):
# MAGIC     if tool_name == "workspace__default__python_exec":
# MAGIC         return DatabricksFunctionClient().execute_function("workspace.default.python_exec", parameters=parameters)
# MAGIC     raise ValueError(f"Unknown tool: {tool_name}")
# MAGIC
# MAGIC def run_agent(prompt):
# MAGIC     """
# MAGIC     Send a user prompt to the LLM, execute any tools if needed,
# MAGIC     and return a list of assistant messages.
# MAGIC     """
# MAGIC     result_msgs = []
# MAGIC
# MAGIC     response = openai_client.chat.completions.create(
# MAGIC         model=LLM_ENDPOINT_NAME,
# MAGIC         messages=[{"role": "user", "content": prompt}],
# MAGIC         tools=builtin_tools,
# MAGIC     )
# MAGIC
# MAGIC     msg = response.choices[0].message
# MAGIC
# MAGIC     if msg.tool_calls:
# MAGIC         # LLM wants to call a tool
# MAGIC         call = msg.tool_calls[0]
# MAGIC         tool_result = call_tool(call.function.name, json.loads(call.function.arguments))
# MAGIC
# MAGIC         try:
# MAGIC             parsed = json.loads(tool_result.value)
# MAGIC             content = parsed.get("result") or parsed.get("stdout") or ""
# MAGIC         except Exception:
# MAGIC             content = tool_result.value
# MAGIC
# MAGIC         result_msgs.append({
# MAGIC             "role": "assistant",
# MAGIC             "content": str(content),
# MAGIC             "name": call.function.name,
# MAGIC             "tool_call_id": call.id
# MAGIC         })
# MAGIC     else:
# MAGIC         # LLM response without tool call
# MAGIC         result_msgs.append(msg.to_dict())
# MAGIC
# MAGIC     return result_msgs
# MAGIC
# MAGIC class QuickstartAgent(ChatAgent):
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         prompt = messages[-1].content
# MAGIC         raw_msgs = run_agent(prompt)
# MAGIC         out = [ChatAgentMessage(id=uuid.uuid4().hex, **m) for m in raw_msgs]
# MAGIC         return ChatAgentResponse(messages=out)
# MAGIC
# MAGIC # Expose agent instance for imports
# MAGIC AGENT = QuickstartAgent()
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC %%writefile quickstart_agent.py
# MAGIC
# MAGIC import json
# MAGIC import uuid
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks_openai import UCFunctionToolkit, DatabricksFunctionClient
# MAGIC from typing import Any, Optional
# MAGIC
# MAGIC import mlflow
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatContext
# MAGIC
# MAGIC # Get an OpenAI client configured to talk to Databricks model serving endpoints
# MAGIC # We'll use this to query an LLM in our agent
# MAGIC openai_client = WorkspaceClient().serving_endpoints.get_open_ai_client()
# MAGIC
# MAGIC # The snippet below tries to pick the first LLM API available in your Databricks workspace
# MAGIC # from a set of candidates. You can override and simplify it
# MAGIC # to just specify LLM_ENDPOINT_NAME.
# MAGIC LLM_ENDPOINT_NAME = None
# MAGIC
# MAGIC def is_endpoint_available(endpoint_name):
# MAGIC   try:
# MAGIC     client = WorkspaceClient().serving_endpoints.get_open_ai_client()
# MAGIC     client.chat.completions.create(model=endpoint_name, messages=[{"role": "user", "content": "What is AI?"}])
# MAGIC     return True
# MAGIC   except Exception:
# MAGIC     return False
# MAGIC   
# MAGIC for candidate_endpoint_name in ["databricks-claude-3-7-sonnet", "databricks-meta-llama-3-3-70b-instruct"]:
# MAGIC     if is_endpoint_available(candidate_endpoint_name):
# MAGIC       LLM_ENDPOINT_NAME = candidate_endpoint_name
# MAGIC assert LLM_ENDPOINT_NAME is not None, "Please specify LLM_ENDPOINT_NAME"
# MAGIC
# MAGIC # Enable automatic tracing of LLM calls
# MAGIC mlflow.openai.autolog()
# MAGIC
# MAGIC # Load Databricks built-in tools (a stateless Python code interpreter tool)
# MAGIC client = DatabricksFunctionClient()
# MAGIC builtin_tools = UCFunctionToolkit(function_names=["system.ai.python_exec"], client=client).tools
# MAGIC for tool in builtin_tools:
# MAGIC     del tool["function"]["strict"]
# MAGIC
# MAGIC def call_tool(tool_name, parameters):
# MAGIC     if tool_name == "system__ai__python_exec":
# MAGIC         return DatabricksFunctionClient().execute_function("system.ai.python_exec", parameters=parameters)
# MAGIC     raise ValueError(f"Unknown tool: {tool_name}")
# MAGIC
# MAGIC def run_agent(prompt):
# MAGIC     """
# MAGIC     Send a user prompt to the LLM, and return a list of LLM response messages
# MAGIC     The LLM is allowed to call the code interpreter tool if needed, to respond to the user
# MAGIC     """
# MAGIC     result_msgs = []
# MAGIC     response = openai_client.chat.completions.create(
# MAGIC         model=LLM_ENDPOINT_NAME,
# MAGIC         messages=[{"role": "user", "content": prompt}],
# MAGIC         tools=builtin_tools,
# MAGIC     )
# MAGIC     msg = response.choices[0].message
# MAGIC     result_msgs.append(msg.to_dict())
# MAGIC
# MAGIC     # If the model executed a tool, call it
# MAGIC     if msg.tool_calls:
# MAGIC         call = msg.tool_calls[0]
# MAGIC         tool_result = call_tool(call.function.name, json.loads(call.function.arguments))
# MAGIC         result_msgs.append({"role": "tool", "content": tool_result.value, "name": call.function.name, "tool_call_id": call.id})
# MAGIC     return result_msgs
# MAGIC
# MAGIC class QuickstartAgent(ChatAgent):
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         prompt = messages[-1].content
# MAGIC         raw_msgs = run_agent(prompt)
# MAGIC         out = []
# MAGIC         for m in raw_msgs:
# MAGIC             out.append(ChatAgentMessage(
# MAGIC                 id=uuid.uuid4().hex,
# MAGIC                 **m
# MAGIC             ))
# MAGIC
# MAGIC         return ChatAgentResponse(messages=out)
# MAGIC
# MAGIC AGENT = QuickstartAgent()
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------



# COMMAND ----------

AGENT = QuickstartAgent()
mlflow.models.set_model(AGENT)


# COMMAND ----------

# MAGIC %md
# MAGIC !mlflow models serve -m "models:/workspace.default.quickstart_agent/8" -p 1234 --no-conda

# COMMAND ----------

# MAGIC %md
# MAGIC import mlflow.pyfunc
# MAGIC
# MAGIC # Use the registered model version
# MAGIC model_uri = "models:/workspace.default.quickstart_agent/8"
# MAGIC
# MAGIC # Serve the model
# MAGIC mlflow.pyfunc.serve_model(model_uri=model_uri, host="0.0.0.0", port=1234)
# MAGIC

# COMMAND ----------



# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

for ep in w.serving_endpoints.list():
    print(ep.name)


# COMMAND ----------

from quickstart_agent import AGENT, LLM_ENDPOINT_NAME
print("LLM endpoint:", LLM_ENDPOINT_NAME)


# COMMAND ----------

import mlflow
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from pkg_resources import get_distribution
from quickstart_agent import AGENT, LLM_ENDPOINT_NAME

# Register the model to the workspace default catalog
catalog_name = spark.sql("SELECT current_catalog()").collect()[0][0]
schema_name = "default"
registered_model_name = f"{catalog_name}.{schema_name}.quickstart_agent"

resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
    DatabricksFunction(function_name="workspace.default.python_exec"),
]

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model=AGENT,  # Pass the instance, not a file path
        extra_pip_requirements=[
            "databricks-openai",  # Add this to ensure MLflow serving has it
            "databricks-sdk",      # Add SDK too
            f"databricks-connect=={get_distribution('databricks-connect').version}"
        ],
        resources=resources,
        registered_model_name=registered_model_name,
    )


# COMMAND ----------

from typing import Any
from pandas import DataFrame

class AgentWrapper(PythonModel):
    def load_context(self, context):
        pass

    def predict(self, context, model_input: Any) -> Any:
        return AGENT.predict(model_input)


# COMMAND ----------

import mlflow

# Use the Unity Catalog model URI
model_uri = "models:/workspace.default.quickstart_agent/14"  # or /latest if you have an alias

# Load the model as a pyfunc
agent_model = mlflow.pyfunc.load_model(model_uri)
print("Model loaded successfully:", agent_model)


# COMMAND ----------

from mlflow.types.agent import ChatAgentMessage

# Load the model
agent_model = mlflow.pyfunc.load_model(model_uri)

# Prepare input
input_dict = {
    "messages": [
        {"id": "1", "role": "user", "content": "Hello, what can you do?"}
    ],
    "context": {},          # Must be a dict, not None
    "custom_inputs": {},    # Must be a dict if included
}

# Call predict
response = agent_model.predict(input_dict)
print(response)


# COMMAND ----------



# COMMAND ----------

model_name = "workspace.default.quickstart_agent"
model_version = 14  # change if needed

print(f"Serving model: {model_name} (v{model_version})")


# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput

w = WorkspaceClient()
endpoint_name = "quickstart-agent-endpoint"

# Look for an existing endpoint
existing = next((ep for ep in w.serving_endpoints.list() if ep.name == endpoint_name), None)

served_entities = [
    ServedEntityInput(
        entity_name="workspace.default.quickstart_agent",
        entity_version="14",
        workload_size="Small",
        scale_to_zero_enabled=True,
    )
]

if existing:
    print(f"🔄 Updating existing endpoint: {endpoint_name}")
    w.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=served_entities
    )
else:
    print(f"🚀 Creating new endpoint: {endpoint_name}")
    w.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            name=endpoint_name,
            served_entities=served_entities
        )
    )

print("✅ Endpoint create/update request sent.")


# COMMAND ----------

# MAGIC %md
# MAGIC mlflow.pyfunc.log_model(
# MAGIC     name="agent",
# MAGIC     python_model=AGENT,
# MAGIC     code_paths=["./quickstart_agent.py"],  # <- package this file
# MAGIC     extra_pip_requirements=[
# MAGIC             "databricks-openai",  # Add this to ensure MLflow serving has it
# MAGIC             "databricks-sdk",      # Add SDK too
# MAGIC             f"databricks-connect=={get_distribution('databricks-connect').version}"
# MAGIC         ],
# MAGIC     resources=resources,
# MAGIC     registered_model_name=registered_model_name,
# MAGIC )
# MAGIC

# COMMAND ----------

logged_agent_info = mlflow.pyfunc.log_model(
    name="agent",
    python_model=AGENT,
    code_paths=["./quickstart_agent.py"],  # <- include your Python file
    extra_pip_requirements=[
        "databricks-openai",
        "databricks-sdk",
        f"databricks-connect=={get_distribution('databricks-connect').version}"
    ],
    resources=resources,
    registered_model_name=registered_model_name,
)


# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput

w = WorkspaceClient()
endpoint_name = "quickstart-agent-endpoint"

# Look for an existing endpoint
existing = next((ep for ep in w.serving_endpoints.list() if ep.name == endpoint_name), None)

served_entities = [
    ServedEntityInput(
        entity_name="workspace.default.quickstart_agent",
        entity_version="14",
        workload_size="Small",
        scale_to_zero_enabled=True,
    )
]

if existing:
    print(f"🔄 Updating existing endpoint: {endpoint_name}")
    w.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=served_entities
    )
else:
    print(f"🚀 Creating new endpoint: {endpoint_name}")
    w.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            name=endpoint_name,
            served_entities=served_entities
        )
    )

print("✅ Endpoint create/update request sent.")


# COMMAND ----------

from time import sleep
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady

w = WorkspaceClient()
endpoint_name = "quickstart-agent-endpoint"

def wait_for_endpoint_ready(name, timeout=600, interval=5):
    elapsed = 0
    while elapsed < timeout:
        ep = next((e for e in w.serving_endpoints.list() if e.name == name), None)
        if ep is None:
            print("Endpoint not found yet...")
        else:
            print(f"Endpoint state: {ep.state}")
            
            if ep.state.ready == EndpointStateReady.READY:
                print("✅ Endpoint is ready!")
                return True
            elif ep.state.ready == EndpointStateReady.FAILED:
                print("❌ Endpoint update failed.")
                return False
            else:
                print("⏳ Endpoint is still updating...")

        sleep(interval)
        elapsed += interval

    print("⚠️ Timeout waiting for endpoint to be ready.")
    return False

# Usage
wait_for_endpoint_ready(endpoint_name)


# COMMAND ----------

 

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
endpoint_name = "quickstart-agent-endpoint"


# COMMAND ----------

DATABRICKS_INSTANCE = "https://dbc-d7920208-a62f.cloud.databricks.com"
TOKEN = "dapifa21789da1af73881af0d4159052ec41"

# COMMAND ----------

import requests
import json

 
endpoint_name = "quickstart-agent-endpoint"
 

endpoint_url = f"{DATABRICKS_INSTANCE}/serving-endpoints/{endpoint_name}/invocations"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "messages": [
        {"role": "user", "content": "Hello, what can you do?", "id": "1"}
    ],
    "context": {},
    "custom_inputs": {}
}

response = requests.post(endpoint_url, headers=headers, data=json.dumps(payload))

print(response.status_code)
try:
    print(response.json())
except json.JSONDecodeError:
    print(response.text)


# COMMAND ----------



# COMMAND ----------

print(logged_agent_info.artifact_path)


# COMMAND ----------

print(model_uri)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC ## Deploy the agent
# MAGIC
# MAGIC Run the cell below to deploy the agent ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/deploy-agent) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/deploy-agent) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/deploy-agent)). Once the agent endpoint starts, you can chat with it via AI Playground ([AWS](https://docs.databricks.com/aws/en/large-language-models/ai-playground) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/large-language-models/ai-playground) | [GCP](https://docs.databricks.com/gcp/en/large-language-models/ai-playground)), or share it with stakeholders ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/review-app) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/review-app) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-evaluation/review-app)) for initial feedback, before sharing it more broadly.

# COMMAND ----------

# MAGIC %md
# MAGIC from quickstart_agent import AGENT
# MAGIC
# MAGIC mlflow.pyfunc.log_model(
# MAGIC     name="agent",
# MAGIC     python_model=AGENT,   # pass the object, not the .py filename
# MAGIC     extra_pip_requirements=[
# MAGIC         f"databricks-connect=={get_distribution('databricks-connect').version}"
# MAGIC     ],
# MAGIC     resources=resources,
# MAGIC     registered_model_name=registered_model_name,
# MAGIC )
# MAGIC

# COMMAND ----------

# TESTING THE MODEL

from mlflow.pyfunc import load_model
from mlflow.types.agent import ChatAgentMessage

# Load the model from MLflow
model_uri = "models:/workspace.default.quickstart_agent/16"
agent = load_model(model_uri)

# Prepare the input message
input_data = {
    "messages": [
        {"role": "user", "content": "What is the square root of 429?"}
    ]
}

# Get the response from the agent
response = agent.predict(input_data)
# print(response)  # Uncomment to see full response

# Extract tool content if available
tool_messages = [m for m in response['messages'] if m['role'] == 'tool']

if tool_messages:
    # Take the last tool message as the answer
    answer = tool_messages[-1]['content'].strip()
else:
    # Fallback: take assistant content if no tool message
    assistant_messages = [m for m in response['messages'] if m['role'] == 'assistant']
    answer = assistant_messages[-1].get('content', '').strip()

print(answer)


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #from databricks import agents
# MAGIC
# MAGIC #deployment_info = agents.deploy(
# MAGIC #    model_name=registered_model_name,
# MAGIC #    model_version=logged_agent_info.registered_model_version,
# MAGIC #)

# COMMAND ----------

from mlflow.pyfunc import load_model
from mlflow.types.agent import ChatAgentMessage

model_uri = "models:/workspace.default.quickstart_agent/16"
agent = load_model(model_uri)

input_data = {
    "messages": [
        {"role": "user", "content": "What is the square root of 429?"}
    ]
}

response = agent.predict(input_data)
#print(response)

# Extract tool content
tool_messages = [m for m in response['messages'] if m['role'] == 'tool']
if tool_messages:
    answer = tool_messages[-1]['content'].strip()  # Take last tool message
else:
    # fallback: take assistant content if no tool message
    assistant_messages = [m for m in response['messages'] if m['role'] == 'assistant']
    answer = assistant_messages[-1].get('content', '').strip()

print(answer)


# COMMAND ----------

import time
time.sleep(10000000)
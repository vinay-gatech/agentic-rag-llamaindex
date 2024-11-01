from dotenv import load_dotenv
from utils import get_doc_tools
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

load_dotenv()
vector_tool, summary_tool = get_doc_tools("metagpt.pdf", "metagpt")
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_tool, summary_tool],
    llm=llm,
    verbose=True
)

agent = AgentRunner(agent_worker)
response = agent.query(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other."
)
print(str(response))

response = agent.chat(
    "Tell me about the evaluation datasets used."
)
print(str(response))
response = agent.chat("Tell me the results over one of the above datasets.")
print(str(response))
agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_tool, summary_tool],
    llm=llm,
    verbose=True
)
agent = AgentRunner(agent_worker)
task = agent.create_task(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other."
)

step_output     = agent.run_step(task.task_id)
completed_steps = agent.get_completed_steps(task.task_id)
print(f"Num completed for task {task.task_id}: {len(completed_steps)}")
print(completed_steps[0].output.sources[0].raw_output)

upcoming_steps = agent.get_upcoming_steps(task.task_id)
print(f"Num upcoming steps for task {task.task_id}: {len(upcoming_steps)}")
print(upcoming_steps[0])

step_output = agent.run_step(
    task.task_id, input="What about how agents share information?"
)
step_output = agent.run_step(task.task_id)
print(step_output.is_last)
response = agent.finalize_response(task.task_id)
print(str(response))
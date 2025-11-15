import ollama


def get_temperature(city: str) -> str:
    """Get the current temperature for a city

    Args:
      city: The name of the city

    Returns:
      The current temperature for the city
    """
    temperatures = {
        "New York": "22°C",
        "London": "15°C",
        "Tokyo": "18°C",
        "Amsterdam": "8°C",

    }
    return temperatures.get(city, "Unknown")


def create_todo_item(title: str, description: str) -> str:
    """Create a new todo item in the database

    Args:
      title: short title of the to do item
      description: long description of the item. Potentially include a smart tip that can help me execute the item. Also, if not trivial, include a first step or even a step plan if it is appropriate given the complexity of the item.

    Returns:
      None
    """
    print(f"title      : {title}\ndescription: {description}")

messages = [{"role": "user", "content": "Herinner me dat ik Soren's zwemlessen stop zet."}]

# pass functions directly as tools in the tools list or as a JSON schema
print(messages)
tools = [get_temperature, create_todo_item]
tools_by_name = {tool.__name__: tool for tool in tools}
response = ollama.chat(model="qwen3:4b", messages=messages, tools=tools, think=True)
# response = ollama.generate(model="qwen3:4b", messages=messages, tools=[get_temperature], think=True)
messages.append(response.message)
if response.message.tool_calls:
    for call in response.message.tool_calls:
        print("calling tool:", call)
        tool = tools_by_name[call.function.name]
        result = tool(**call.function.arguments)
        # result = get_temperature(**call.function.arguments)
        messages.append({"role": "tool", "tool_name": call.function.name, "arguments":call.function.arguments, "content": str(result)})

    print(messages)

    final_response = ollama.chat(model="qwen3:4b", messages=messages, tools=[get_temperature], think=True)
    print(final_response.message.content)
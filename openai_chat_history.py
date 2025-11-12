from openai import OpenAI
from dotenv import load_dotenv
import os
import gradio as gr
import json


load_dotenv(".env")
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("✅ API Key loaded successfully.\n")
else:
    print("❌ Failed to load API Key.")
    exit()

client = OpenAI(api_key=api_key)

#Based on user destination input, return fare amount
def travel_ticket_fare_calculator(destination_city: list[str]):
    result = {}
    ticket_fare = {
        "chennai": {
            "price": 1000,
            "currency": "INR"
        },
        "bangalore": {
            "price": 1200,
            "currency": "INR"
        },
        "mumbai": {
            "price": 1500,
            "currency": "INR"
        }
    }
    for city in destination_city:
        city_lower = city.lower()
        if city_lower in ticket_fare:
            result = ticket_fare[city_lower]
        else:
            result[city_lower] = {"error": "The given city is not found in the database"}
    return result

#Define function declarations to the model
ticket_calculator_function = {
    "name": "check_price_of_ticket",
    "description": "This function returns the ticket fare based on the given city",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "The list of cities which user wants to know the ticket fare"
        }
    },
    "required": ["destination_city"]
}
}

system_instruction = "You're an travel advisor and you only focus to answer for questions related to travel."
system_instruction += "When user asks for one way ticket fare you can call the required tool from tool declarations, otherwise if that is a general query, response"
"should be given normally and if the query is not related to tarvel, please respond like couldn't able to answer other than travelling related questions."

def chat_with_model_with_history(query, history):
    if history == []:
        history.append(
            {
                "role": "system",
                "content": system_instruction
            }
        )
    user_query = []
    user_query.append({"role": "user", "content": query})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history + user_query,
        functions=[ticket_calculator_function],
        function_call="auto"
    )

    if response.choices[0].message.function_call:
        print("Function call detected...")
        function_name = response.choices[0].message.function_call.name
        function_args = json.loads(response.choices[0].message.function_call.arguments)
        print(function_name, function_args)
        if function_name == "check_price_of_ticket":
            destination_city = function_args.__getitem__("destination_city")
            response = travel_ticket_fare_calculator(destination_city=destination_city)
            print(response)
            ai_response_raw = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=history + user_query + [
                    {
                        "role": "function",
                        "name": function_name,
                        "content": f"Ticket fare details for city {destination_city} is {response.get('price')}"
                    }
                ]
            )
            ai_response = ai_response_raw.choices[0].message.content
        else:
            print(f"Function {function_name} is not recognized.")
    else:
        ai_response = response.choices[0].message.content
    return ai_response

view = gr.ChatInterface(
    fn=chat_with_model_with_history,
    type="messages",
    title="💬 Chat with Your Friendly AI Tutor (with History)",
    description="A chat interface that maintains conversation history with the GPT-4o-mini model."
)
view.launch()
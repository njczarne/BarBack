import os
import openai
import json

# Set your OpenAI API key securely, ideally via environment variable
openai.api_key = ""

# This function should be explicitly called to process the menu and handle the OpenAI API call
def menu_change(current_drink, input_change):
    
    raw_input = f"Current Drink: {current_drink}, Request: {input_change}"
    print(f"CHange: {input_change}")
    
    # Load menu data
    with open("ingredients.json", "r") as f:
        drink_menu = json.load(f)
    
    menu_prompt = json.dumps(drink_menu, indent=2)
    system_message = "You are a bartender assistant AI. You know a drink menu and can customize recipes based on user requests."

    user_prompt = f"""
    Here is the drink menu:

    {menu_prompt}

    A customer said:
    "{raw_input}"

    Instructions:
    - Fix any speech-to-text errors in the input.
    - Identify the requested drink.
    - Apply the user's customization preferences (like 'no salt' or 'extra lime').
    - Return a cleaned version of the drink order and a final list of ingredients.
    - Only use drinks that exist in the menu.
    - Refer Cranberry juice as "Cran Juice"
    
    Only use these ingredients (Do not add any other ingredients, even if the user asks for it):
    Rum, Tequila, Soda water, Coke, Lime juice, Cranberry juice
    
    Common Transciption errors
    Soda water: "Sort of water", "So the water"
    Coke: "Cook", "Code"
    Lime: "Line"
    Rum: "Run", "Room", "Ramen"
    Cran Juice: "Fran Jews", "Frank Use"
    Cran: "Creme", "Crumbly"

    Respond in this format:
    Do not any additional phrases to "drink_name" (even if you make changes. dont add the word "customized")
    Cleaned: <fixed order>
    Order:
    {{
      "name": "drink_name",
      "recipe": ["..."]
    }}
    """

    # Make the API call to OpenAI's ChatCompletion
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Ensure you'r  e using the correct model name
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    
    content = response['choices'][0]['message']['content']
    
    try:
        json_start = content.index("{")
        order_json = content[json_start:]
        order_data = json.loads(order_json)
        
        with open("order.json", "w") as f:
            json.dump(order_data, f, indent=4)
            print("Reached")
    except Exception as e:
        json.dump("Error with customization", f, indent=4)
    

    # Return the response content
    return response['choices'][0]['message']['content']

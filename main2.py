import time
import threading
from menu_obj import SmartBartender
from LLM_menu import menu_change
from voice import listen_until_ending, listen_for_phrase
import json
import os
from pump_code import pump_drink, pump_timing

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Global variable to store the current drink name
current_drink_name = None
ACTIVATION_PHRASE = "wake up"
no_custom = True

# Function to run the AI process in the background
def ai_task():
    global current_drink_name
    while True:
        # Wait 10 seconds before processing the AI task
        print(f" Waiting for activation phrase: {ACTIVATION_PHRASE}...")
        result = listen_for_phrase(ACTIVATION_PHRASE)
        print("✅ Activation Detected:", result)

        # Listen for commands until ending phrase
        output = listen_until_ending()
        
        # If we have a current drink, send it to the AI
        if current_drink_name:
            print(f"Processing AI for drink: {current_drink_name}")
            change = menu_change(current_drink_name, output)  # Pass the current drink to AI
            print(f"AI change result: {change}")
            
            bartender.display_order()
            no_custom = False
            
        else:
            print("No current drink selected yet.")

# Create an instance of the SmartBartender class
bartender = SmartBartender()

# Start the AI task in a separate thread
ai_thread = threading.Thread(target=ai_task)
ai_thread.daemon = True  # This ensures the thread will exit when the main program ends
ai_thread.start()

print("test")

# Main loop to keep checking the current drink while the menu runs
while True:
    current_drink_name = bartender.get_current_drink_name()

    if bartender.buttonR.value:
        bartender.display_mode = "menu"
        bartender.next_drink()
        bartender.phase = "first"

    elif bartender.buttonL.value:
        bartender.display_mode = "menu"
        bartender.prev_drink()
        bartender.phase = "first"

    elif bartender.buttonC.value:
        drink_name = bartender.get_current_drink_name()
        
        if bartender.phase == "first":
            if no_custom:
                try:
                    with open("ingredients.json", "r") as f:
                        all_recipes = json.load(f)

                    current_recipe = all_recipes.get(drink_name, [])

                    # Structure to include name and recipe
                    order_data = {
                        "name": drink_name,
                        "recipe": current_recipe
                    }

                    try:
                        with open("order.json", "w") as f:
                            json.dump(order_data, f, indent=4)
                    except Exception as e:
                        print("Error writing order:", e)

                    bartender.display_order()

                except Exception as e:
                    print("Error reading recipe:", e)
                    
        elif bartender.phase == "second":
            
            with open("order.json", "r") as f:
                current_drink = json.load(f)
            
            print("Making Drink")
            recipe = current_drink.get("recipe", [])
            int_ingredients = pump_timing(recipe)
            pump_drink(int_ingredients[0],  # rum
                int_ingredients[1],  # tequila
                int_ingredients[2],  # lime
                int_ingredients[3],  # coke
                int_ingredients[4],  # cran
                int_ingredients[5]   # soda
            )
            bartender.phase = "first"
            bartender.display_drink(0)
                

    # Only redraw drink if in 'menu' mode
    if bartender.display_mode == "menu":
        bartender.display_drink(bartender.current_index)

    time.sleep(0.1)

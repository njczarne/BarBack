import time
import threading
# --- Import the setup/teardown functions ---
from voice_thread2 import (
    listen_until_ending,
    listen_for_phrase,
    initialize_audio_system,
    start_worker,
    stop_worker,
    cleanup_temp_files
)
# -------------------------------------------
from menu_obj import SmartBartender
from LLM_menu import menu_change
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
        # Wait 10 seconds before processing the AI task # REMOVED - This wait seems unnecessary now
        print(f" Waiting for activation phrase: {ACTIVATION_PHRASE}...")
        # This call should now work because the worker is started below
        activated = listen_for_phrase(ACTIVATION_PHRASE)

        # Check the return value before printing/proceeding
        if not activated:
             print("Activation listening stopped or failed.")
             # Decide if the thread should exit or retry
             # For now, let's retry after a delay
             time.sleep(1)
             continue # Skip the rest of the loop and wait for activation again

        # If activated is True, proceed
        # print("✅ Activation Detected:", activated) # Optional: Simplified confirmation already printed by listen_for_phrase

        # Listen for commands until ending phrase
        output = listen_until_ending("thank you")

        if output is None:
            print("Command listening stopped or failed.")
            # Decide if the thread should exit or retry
            time.sleep(1)
            continue # Skip processing and wait for activation again

        # Print the heard command - listen_until_ending no longer prints this
        print(f"Commands heard by AI Task: {output}")

        # If we have a current drink, send it to the AI
        if current_drink_name:
            print(f"Processing AI for drink: {current_drink_name}")
            change = menu_change(current_drink_name, output)  # Pass the current drink to AI
            print(f"AI change result: {change}")

            bartender.display_order()
            no_custom = False

        else:
            print("No current drink selected yet for AI processing.")

# --- Initialize Audio System and Start Worker ---
print("Initializing audio system...")
if not initialize_audio_system():
    print("Failed to initialize audio system. Exiting.")
    exit(1)

print("Starting voice worker...")
start_worker()
# -------------------------------------------------

# Create an instance of the SmartBartender class
bartender = SmartBartender()

# Start the AI task in a separate thread
ai_thread = threading.Thread(target=ai_task)
ai_thread.daemon = True  # This ensures the thread will exit when the main program ends
ai_thread.start()

# print("test") # Original test print

# --- Main loop with Shutdown Handling ---
try:
    while True:
        current_drink_name = bartender.get_current_drink_name()

        if bartender.buttonR.value:
            bartender.display_mode = "menu"
            bartender.next_drink()
            bartender.phase = "first"
            no_custom = True # Reset customization on drink change

        elif bartender.buttonL.value:
            bartender.display_mode = "menu"
            bartender.prev_drink()
            bartender.phase = "first"
            no_custom = True # Reset customization on drink change

        elif bartender.buttonC.value:
            drink_name = bartender.get_current_drink_name()

            if bartender.phase == "first":
                if no_custom:
                    try:
                        with open("ingredients.json", "r") as f:
                            all_recipes = json.load(f)

                        current_recipe = all_recipes.get(drink_name, [])

                        order_data = {
                            "name": drink_name,
                            "recipe": current_recipe
                        }

                        try:
                            with open("order.json", "w") as f:
                                json.dump(order_data, f, indent=4)
                        except Exception as e:
                            print("Error writing order:", e)

                        bartender.display_order() # Changes phase to "second" inside

                    except Exception as e:
                        print("Error reading recipe:", e)

            elif bartender.phase == "second":

                try: # Add error handling for reading order.json
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
                    bartender.display_drink(0) # Go back to first drink view? Or current index?
                    no_custom = True # Reset customization after making drink

                except FileNotFoundError:
                    print("Error: order.json not found. Cannot make drink.")
                    bartender.phase = "first" # Reset phase
                    bartender.display_drink(bartender.current_index) # Show current drink again
                except Exception as e:
                    print(f"Error processing order or pumping drink: {e}")
                    bartender.phase = "first" # Reset phase

        # Only redraw drink if in 'menu' mode (or maybe always redraw?)
        # Let's assume display_drink/display_order handles drawing
        # if bartender.display_mode == "menu":
        #     bartender.display_drink(bartender.current_index)

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nCtrl+C detected. Shutting down bartender.")
finally:
    # --- Stop Worker and Cleanup ---
    print("Stopping voice worker and cleaning up...")
    stop_worker()
    cleanup_temp_files()
    print("Bartender application finished.")
    # -------------------------------
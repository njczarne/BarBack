from gpiozero import OutputDevice
from time import sleep
import threading

# GPIO19 is pin 35 (BCM mode)
rum_pump = OutputDevice(19, active_high=False, initial_value=False)
teq_pump = OutputDevice(26, active_high=False, initial_value=False)
lime_pump = OutputDevice(12, active_high=False, initial_value=False)
coke_pump = OutputDevice(16, active_high=False, initial_value=False)
cran_pump = OutputDevice(20, active_high=False, initial_value=False)
soda_pump = OutputDevice(21, active_high=False, initial_value=False)


def load_pump():
    # load time is 2.2165 seconds
    time = 2.2165
    # time = 5
    threads = []
    
    threads.append(threading.Thread(target=run_pump, args=(rum_pump, time)))
    threads.append(threading.Thread(target=run_pump, args=(teq_pump, time)))
    threads.append(threading.Thread(target=run_pump, args=(lime_pump, time)))
    threads.append(threading.Thread(target=run_pump, args=(coke_pump, time)))
    threads.append(threading.Thread(target=run_pump, args=(cran_pump, time)))
    threads.append(threading.Thread(target=run_pump, args=(soda_pump, time)))
    
    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()
        
    print("Pumps Loaded")
    
def pump_timing(recipe):
    # Define the ingredient order you're tracking
    tracked_ingredients = ["Rum", "Tequila", "Lime juice", "Coke", "Cran juice", "Soda water"]
    
    # Initialize the result with 0s
    result = [0] * len(tracked_ingredients)

    # Go through each item in the recipe list
    for item in recipe:
        for ingredient, amount in item.items():
            if ingredient in tracked_ingredients:
                index = tracked_ingredients.index(ingredient)
                result[index] = amount * 16  # Multiply the actual amount by 2

    return result


def run_pump(pump, duration):
    pump.on()
    sleep(duration)
    pump.off()

def pump_drink(rum, tequila, lime, coke, cran, soda):
    print("Dispensing all liquids simultaneously...")

    threads = []

    if rum != 0:
        threads.append(threading.Thread(target=run_pump, args=(rum_pump, rum)))

    if tequila != 0:
        threads.append(threading.Thread(target=run_pump, args=(teq_pump, tequila)))

    if lime != 0:
        threads.append(threading.Thread(target=run_pump, args=(lime_pump, lime)))

    if coke != 0:
        threads.append(threading.Thread(target=run_pump, args=(coke_pump, coke)))

    if cran != 0:
        threads.append(threading.Thread(target=run_pump, args=(cran_pump, cran)))

    if soda != 0:
        threads.append(threading.Thread(target=run_pump, args=(soda_pump, soda)))

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    print("Done.")


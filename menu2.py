from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306
import board
import busio
import time
import json
import digitalio

# Setup Buttons
buttonR = digitalio.DigitalInOut(board.D23)
buttonR.direction = digitalio.Direction.INPUT
buttonR.pull = digitalio.Pull.DOWN

buttonL = digitalio.DigitalInOut(board.D24)
buttonL.direction = digitalio.Direction.INPUT
buttonL.pull = digitalio.Pull.DOWN

# Setup OLED
i2c = busio.I2C(board.SCL, board.SDA)
oled = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c)

oled.fill(0)
oled.show()

image = Image.new("1", (oled.width, oled.height))
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

MAX_WIDTH = 125
LINE_HEIGHT = 10

def load_menu(filename='menu.json'):
    with open(filename, 'r') as f:
        return json.load(f)
    
def load_ingr(filename='ingredients.json'):
    with open(filename, 'r') as f:
        return json.load(f)
    
def wrap_text(text, max_width, font):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + word + " "
        bbox = draw.textbbox((0, 0), test_line, font=font)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    lines.append(current_line.strip())
    return lines

def display_drink(index):
    current_drink = list(menu.keys())[index]
    current_ingredients = menu[current_drink]
    
    draw.rectangle((0, 0, oled.width, oled.height), outline=0, fill=0)
    
    y = 0
    drink_lines = wrap_text(current_drink, MAX_WIDTH, font)
    for line in drink_lines:
        draw.text((5, y), line, font=font, fill=255)
        y += LINE_HEIGHT
    draw.line((0, y, oled.width, y), fill=255)
    y += 4
    
    draw.text((5, y), "Ingredients:", font=font, fill=255)
    y += LINE_HEIGHT
    
    ingredients_text = ', '.join(current_ingredients)
    ingredient_lines = wrap_text(ingredients_text, MAX_WIDTH, font)
    for line in ingredient_lines:
        if y > oled.height - LINE_HEIGHT:
            break
        draw.text((5, y), line, font=font, fill=255)
        y += LINE_HEIGHT

    oled.image(image)
    oled.show()
    
    return current_drink

# Load menu
menu = load_menu()
x = 0
display_drink(x)

try:
    while True:
        # Check Right Button
        if buttonR.value:
            x = (x + 1) % len(menu)
            print(f"Right Button Pressed! Drink index: {x}")
            display_drink(x)
            # Wait for button release
            while buttonR.value:
                time.sleep(0.05)

        # Check Left Button
        if buttonL.value:
            x = (x - 1) % len(menu)
            print(f"Left Button Pressed! Drink index: {x}")
            display_drink(x)
            # Wait for button release
            while buttonL.value:
                time.sleep(0.05)

        time.sleep(0.05)  # Small loop delay

except KeyboardInterrupt:
    print("Program stopped.")

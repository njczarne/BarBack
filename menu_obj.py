import os
import time
import json
import digitalio
import busio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306
import board


class SmartBartender:
    def __init__(self, menu_filename='menu.json', ingr_filename='ingredients.json'):
        # Setup Buttons
        self.buttonR = digitalio.DigitalInOut(board.D23)
        self.buttonR.direction = digitalio.Direction.INPUT
        self.buttonR.pull = digitalio.Pull.DOWN

        self.buttonL = digitalio.DigitalInOut(board.D24)
        self.buttonL.direction = digitalio.Direction.INPUT
        self.buttonL.pull = digitalio.Pull.DOWN
        
        self.buttonC = digitalio.DigitalInOut(board.D25)
        self.buttonC.direction = digitalio.Direction.INPUT
        self.buttonC.pull = digitalio.Pull.DOWN

        # Setup OLED
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.oled = adafruit_ssd1306.SSD1306_I2C(128, 64, self.i2c)

        self.oled.fill(0)
        self.oled.show()

        self.image = Image.new("1", (self.oled.width, self.oled.height))
        self.draw = ImageDraw.Draw(self.image)

        # Load fonts
        self.font = ImageFont.load_default()
        self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)

        self.MAX_WIDTH = 125
        self.LINE_HEIGHT = 10

        # Load menu and ingredients data
        self.menu = self.load_menu(menu_filename)
        self.ingredients = self.load_ingr(ingr_filename)

        self.current_index = 0
        self.display_drink(self.current_index)
        
        self.display_mode = "menu"
        self.phase = "first"

    def load_menu(self, filename='menu.json'):
        with open(filename, 'r') as f:
            return json.load(f)

    def load_ingr(self, filename='ingredients.json'):
        with open(filename, 'r') as f:
            return json.load(f)

    def wrap_text(self, text, max_width, font):
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + word + " "
            bbox = self.draw.textbbox((0, 0), test_line, font=font)
            width = bbox[2] - bbox[0]
            if width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        lines.append(current_line.strip())
        return lines

    def display_drink(self, index):
        current_drink = list(self.menu.keys())[index]
        current_ingredients = self.menu[current_drink]

        self.draw.rectangle((0, 0, self.oled.width, self.oled.height), outline=0, fill=0)

        y = 0
        drink_lines = self.wrap_text(current_drink, self.MAX_WIDTH, self.font)
        for line in drink_lines:
            self.draw.text((5, y), line, font=self.font, fill=255)
            y += self.LINE_HEIGHT
        self.draw.line((0, y, self.oled.width, y), fill=255)
        y += 4

        self.draw.text((5, y), "Ingredients:", font=self.font, fill=255)
        y += self.LINE_HEIGHT

        ingredients_text = ', '.join(current_ingredients)
        ingredient_lines = self.wrap_text(ingredients_text, self.MAX_WIDTH, self.font)
        for line in ingredient_lines:
            if y > self.oled.height - self.LINE_HEIGHT:
                break
            self.draw.text((5, y), line, font=self.font, fill=255)
            y += self.LINE_HEIGHT

        self.oled.image(self.image)
        self.oled.show()

        self.current_drink_name = current_drink

    def next_drink(self):
        self.current_index = (self.current_index + 1) % len(self.menu)
        self.display_drink(self.current_index)

    def prev_drink(self):
        self.current_index = (self.current_index - 1) % len(self.menu)
        self.display_drink(self.current_index)

    def get_current_drink_name(self):
        return getattr(self, 'current_drink_name', None)

    def display_order(self):
        self.display_mode = "confirm"
        self.phase = "second"

        self.draw.rectangle((0, 0, self.oled.width, self.oled.height), outline=0, fill=0)

        try:
            with open("order.json", "r") as f:
                order_data = json.load(f)

            drink_name = order_data.get("name", "Unknown")
            recipe = order_data.get("recipe", [])

            font_default = self.font_small
            font_small = self.font_small

            def get_lines(font):
                drink_lines = self.wrap_text(drink_name, self.MAX_WIDTH, font)
                ingredient_lines = []
                for item in recipe:
                    for ingredient, amount in item.items():
                        line = f"{ingredient}: {amount} oz"
                        ingredient_lines.extend(self.wrap_text(line, self.MAX_WIDTH, font))
                return drink_lines, ingredient_lines

            # First try default font
            font = font_default
            line_height = 10
            drink_lines, ingredient_lines = get_lines(font)
            total_lines = len(drink_lines) + len(ingredient_lines)

            # Reserve exact space for header
            header_height = line_height + 2 + 2  # "Confirm Change" + line + spacing
            available_height = self.oled.height - header_height

            # Check if it fits, else try small font
            if total_lines * line_height > available_height:
                font = font_small
                line_height = 8
                drink_lines, ingredient_lines = get_lines(font)
                total_lines = len(drink_lines) + len(ingredient_lines)
                header_height = line_height + 2 + 2
                available_height = self.oled.height - header_height

            # If still doesn't fit, shrink line_height further to 7 or 6
            if total_lines * line_height > available_height:
                line_height = available_height // total_lines

            # Draw header
            y = 0
            self.draw.text((5, y), "Confirm Change:", font=font, fill=255)
            y += line_height + 2
            self.draw.line((0, y, self.oled.width, y), fill=255)
            y += 2

            # Draw drink name
            for line in drink_lines:
                if y + line_height > self.oled.height:
                    break
                self.draw.text((5, y), line, font=font, fill=255)
                y += line_height

            # Draw ingredients
            for line in ingredient_lines:
                if y + line_height > self.oled.height:
                    break
                self.draw.text((5, y), line, font=font, fill=255)
                y += line_height

        except Exception as e:
            y = 0
            self.draw.text((5, y), "Error loading order", font=self.font, fill=255)
            y += 10
            self.draw.text((5, y), "Try Voice Customization Again", font=self.font, fill=255)

        self.oled.image(self.image)
        self.oled.show()
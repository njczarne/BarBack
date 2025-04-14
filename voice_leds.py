from gpiozero import PWMLED
from time import sleep

led = PWMLED(17)  # GPIO17 = pin 11

def turn_on_led():
    led.pulse(fade_in_time=1, fade_out_time=1, n=None, background=True)
    # `n=None` makes it pulse indefinitely
    # `background=True` allows your program to continue while pulsing

def turn_off_led():
    led.off()

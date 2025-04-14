
from gpiozero import OutputDevice
from time import sleep


rum_pump = OutputDevice(19, active_high=False, initial_value=False)
teq_pump = OutputDevice(26, active_high=False, initial_value=False)
lime_pump = OutputDevice(12, active_high=False, initial_value=False)
coke_pump = OutputDevice(16, active_high=False, initial_value=False)
cran_pump = OutputDevice(20, active_high=False, initial_value=False)
soda_pump = OutputDevice(21, active_high=False, initial_value=False)

cran_pump.on()

sleep(16)

cran_pump.off()



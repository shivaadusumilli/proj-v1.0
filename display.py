#git link -> https://github.com/the-raspberry-pi-guy/lcd
import lcddriver
import time
display = lcddriver.lcd()

def updateDisplay():
    LogP("updating display")
    display.lcd_display_string(L1, 1) # Write line of text to first line of display
    display.lcd_display_string(L2, 2) # Write line of text to second line of display



# Main body of code
try:
    while True:
        # Remember that your sentences can only be 16 characters long!
        print("Writing to display")
        display.lcd_display_string("Greetings Human!", 1) # Write line of text to first line of display
        display.lcd_display_string("Demo Pi Guy code", 2) # Write line of text to second line of display
        time.sleep(2)                                     # Give time for the message to be read
        display.lcd_display_string("I am a display!", 1)  # Refresh the first line of display with a different message
        time.sleep(2)                                     # Give time for the message to be read
        display.lcd_clear()                               # Clear the display of any data
        time.sleep(2)                                     # Give time for the message to be read

except KeyboardInterrupt: # If there is a KeyboardInterrupt (when you press ctrl+c), exit the program and cleanup
    print("Cleaning up!")
    display.lcd_clear()

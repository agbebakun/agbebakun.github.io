RED_HSV_LOWER = [0, 100, 100]
RED_HSV_UPPER = [20, 255, 255]
RED_RGB = (0, 0, 255)

GREEN_HSV_LOWER = [36, 0, 0]
GREEN_HSV_UPPER = [86, 255, 255]
GREEN_RGB = (0, 255, 0)

BLUE_HSV_LOWER = [100, 60, 60]
BLUE_HSV_UPPER = [140, 255, 255]
BLUE_RGB = (255, 0, 0)

YELLOW_RGB = (0, 255, 255)
WHITE_RGB = (255, 255, 255)

# alternative classes
USE_ALT_CLASSES = ""  # "" "same" "test" "test2" "many" "all"

if USE_ALT_CLASSES == None or USE_ALT_CLASSES == "":
    # default classes (used for web)
    CLASSES = ["apple", "book", "bowtie", "candle", "cloud", "cup", "door", "envelope", "eyeglasses", "guitar", "hammer",
            "hat", "ice cream", "leaf", "scissors", "star", "t-shirt", "pants", "lightning", "tree"]
    
elif USE_ALT_CLASSES == "same":
    # same as default classes (to test retraining)
    CLASSES = ["apple", "book", "bowtie", "candle", "cloud", "cup", "door", "envelope", "eyeglasses", "guitar", "hammer",
            "hat", "ice cream", "leaf", "scissors", "star", "t-shirt", "pants", "lightning", "tree"]

elif USE_ALT_CLASSES == "test":
    # test
    CLASSES = [
        "cloud", "bush", "leaf", "feather", "snake", "river", "cell phone", "remote control",
    ]

elif USE_ALT_CLASSES == "test2":
    # test2
    CLASSES = [
        "snake", "river",
    ]

elif USE_ALT_CLASSES == "many":
    # too many classes?
    CLASSES = [
        "airplane", "arm", "bear", "bed", "bench", "brain", "bush", "bus", "calculator", "calendar", 
        "camera", "cell phone", "chair", "clock", "cloud", "computer", "couch", "cow", "dog", "drill", 
        "eye", "face", "feather", "finger", "flashlight", "floor lamp", "hamburger", "headphones", "helicopter", 
        "horse", "hospital", "keyboard", "key", "laptop", "leaf", "leg", "light bulb", "lightning", "map", 
        "moon", "motorbike", "paintbrush", "pencil", "piano", "remote control", "river", "sailboat", "sandwich", 
        "screwdriver", "snake", "speedboat", "spreadsheet", "star", "stethoscope", "streetlight", "sun", "sword", 
        "telephone", "television", "toe", "tree", "truck"
    ]

elif USE_ALT_CLASSES == "all":
    # all data
    CLASSES = [
        "airplane", "apple", "arm", "bear", "bed", "bench", "book", "bowtie", "brain", "bus", 
        "bush", "calculator", "calendar", "camera", "candle", "cell phone", "chair", "clock", "cloud", "computer", 
        "couch", "cow", "cup", "dog", "door", "drill", "envelope", "eye", "eyeglasses", "face", 
        "feather", "finger", "flashlight", "floor lamp", "guitar", "hamburger", "hammer", "hat", "headphones", "helicopter", 
        "horse", "hospital", "ice cream", "key", "keyboard", "knife", "laptop", "leaf", "leg", "light bulb", 
        "lightning", "map", "moon", "motorbike", "paintbrush", "pants", "pencil", "piano", "remote control", "river", 
        "sailboat", "sandwich", "scissors", "screwdriver", "snake", "speedboat", "spreadsheet", "star", "stethoscope", "streetlight", 
        "sun", "sword", "syringe", "t-shirt", "telephone", "television", "toe", "tree", "truck"
    ]

else:
    raise ValueError("Invalid USE_ALT_CLASSES value")

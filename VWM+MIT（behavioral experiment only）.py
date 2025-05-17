from psychopy import visual, core, event, data, gui, monitors
import random
import pandas as pd
import math  # Import math for trigon1ometric functions
import sys
import os
import numpy as np

# Participant ID
exp_info = {'Participant': ''}
dlg = gui.DlgFromDict(dictionary=exp_info, title='VWM Experiment')
if not dlg.OK:
    core.quit()

# Screen parameters
monitor_name = "testMonitor"
my_monitor = monitors.Monitor(monitor_name)
my_monitor.setSizePix([1920, 1080])
my_monitor.setWidth(31)  # Approximate screen width in cm
my_monitor.setDistance(57)

# Experiment background screen
win = visual.Window(
    size=[1920, 1080],
    monitor=my_monitor,
    color="white",
    units="deg",
    fullscr=True
)

# Unit conversion: deg
screen_width_deg = 31  # Approximate screen width in degrees
stimulus_radius = (screen_width_deg / 4) - 1  # Reduce distance from center by 1 cm

# Load MIT images
image_files = [
    r"stimuli\1.jpg",
    r"stimuli\2.jpg",
    r"stimuli\3.jpg",
    r"stimuli\4.jpg",
    r"stimuli\5.jpg",
    r"stimuli\6.jpg"
]
images = [visual.ImageStim(win, image=img, units='deg') for img in image_files]
padding = 0.01 

# Define fixation
def Fixation():
    fixation = visual.TextStim(win, text="+", color="black", height=1, pos=(0, 0))
    fixation.draw()
    win.flip()
    core.wait(0.5)

# Define retention interval
def Retention_Interval():
    fixation = visual.TextStim(win, text="+", color="black", height=1, pos=(0, 0))
    fixation.draw()
    win.flip()
    core.wait(1)

# Define trial type
def Trial_begin(label, duration=2.0):
    number = visual.TextStim(win, text=label, color="red", height=1, pos=(0, 0))
    number.draw()
    win.flip()
    core.wait(duration)

# Save data and exit
def save_data_and_exit():
    os.makedirs("data", exist_ok=True)
    filename = f"data/{exp_info['Participant']}_partial"
    df = pd.DataFrame(data_list)
    df.to_excel(filename + ".xlsx", index=False)
    df.to_csv(filename + ".csv", index=False)
    print("Partial data saved")
    win.close()
    core.quit()

# Confirm task understanding
def confirm_understanding():
    mouse = event.Mouse()
    rt_clock = core.Clock()

    instruction = (
        "Do you understand this task procedure?\n\n"
        "YES: Click Left mouse button,\n"
        "You will proceed to practicing this task without instructions,\n"
        "but still need to response to the task.\n\n"
        "NO: Click Right mouse button,\n"
        "You will repeat this practice with instrcution.\n\n"
    )
    text_stim = visual.TextStim(win, text=instruction, color='black', height=0.7, wrapWidth=25)
    win.flip()
    text_stim.draw()
    win.flip()

    while True:
        buttons = mouse.getPressed()
        keys = event.getKeys()

        if buttons[0]:  # Left mouse button
            return "continue"  # Continue to formal experiment
        elif buttons[2]:  # Right mouse button
            return "repeat"  # Return to practice loop
        elif "escape" in keys:
            print("🚪 Participant exited mid-experiment, saving data so far")
            save_data_and_exit()
        else:
            pass  # ignore other keys
    core.wait(0.1)

#Feedback during practice phase
def show_feedback(vwm_correct=None, mit_correct=None):
    feedback_text = ""  # ✅ Initialize variable 

    if vwm_correct is not None and mit_correct is not None:
        # Combined task feedback
        feedback_text += f"Task 1: {'Correct ✅' if vwm_correct else 'Wrong ❌'}\n"
        feedback_text += f"Task 2: {'Correct ✅' if mit_correct else 'Wrong ❌'}"
    elif vwm_correct is not None:
        feedback_text += f"{'Correct ✅' if vwm_correct else 'Wrong ❌'}"
    elif mit_correct is not None:
        feedback_text += f"{'Correct ✅' if mit_correct else 'Wrong ❌'}"
    
    feedback = visual.TextStim(win, text=feedback_text, color="black", height=0.8, wrapWidth=20)
    feedback.draw()
    win.flip()
    core.wait(1.5)  # Display for 1.5 seconds

# Instructions
class Instructions:
    def __init__(self, win):
        self.win = win
        self.mouse = event.Mouse()

    # Click the mouse
    def wait_for_mouse(self, timeout=None):
        timer = core.Clock()
    
         # Step 1: wait until mouse is released
        while any(self.mouse.getPressed()):
            core.wait(0.05)
        # Step 2: wait for the next new click
        while True:
            if any(self.mouse.getPressed()):
                break
            elif "escape" in event.getKeys():
                save_data_and_exit()
            elif timeout and timer.getTime() >= timeout:
                break
            core.wait(0.1)

    # 1. Show experiment start instructions 
    def begin_instructions(self):
        instructions = visual.TextStim(
            self.win,
            text=(
                "Welcome to the experiment!\n\n"
                "The experiment will take around 30 minutes in total.\n"
                "It includes three different tasks.\n"
                "Before the formal experiment begins, you will first complete some practice trials "
                "until you fully understand each task.\n\n"
                "You’ll get feedback in practice, but not in the formal task.\n"
                "Click the left or right mouse button to start the practice."
            ),
            color="black", height=0.7, wrapWidth=25, pos=(0, 0)
        )
        instructions.draw()
        self.win.flip()

        while True:
            if any(self.mouse.getPressed()):
                break
            elif "escape" in event.getKeys():
                save_data_and_exit()
            core.wait(0.1)

    # 2. Show VWM instructions
    def VWM_show_instructions(self):
        instructions = visual.TextStim(self.win, text="Task 1\n\nYou need to remember the color and the location of three circles\nClick Mouse button (left or right) to start.", 
                                       color="black", height=0.7, wrapWidth=25, pos=(0, 0))
        instructions.draw()
        self.win.flip()
        self.wait_for_mouse()
    
    # 3.Show MIT instructions
    def MIT_show_instructions(self):
        instruction_text = visual.TextStim(
            self.win,
            text=(
                "Task 2\n\n"
                "The objects with borders (targets) are the ones you need to track.\n\n"
                "Once the borders disappear, everything will start moving — "
                "just focus on your targets and remember where they go.\n\n"
                "When the motion stops, one object will be highlighted. "
                "Was it one of your targets? If yes, which one was it?\n\n"
                "Click the mouse button (left or right) to start."
            ),
            color="black", height=0.7, wrapWidth=25, pos=(0, 0)
        )
        instruction_text.draw()
        self.win.flip()
        self.wait_for_mouse()
    
    # 4.Show VWM+MIT instructions
    def VWM_MIT_show_instructions(self):
        instruction_text = visual.TextStim(
            self.win,
            text=(
                "Task 1 + 2\n\n"
                "Now you will perform both of the previous tasks at the same time.\n\n"
                "First, remember the colors and positions of the three circles from Task 1.\n"
                "Then, track the targets from Task 2 and remember where they go.\n\n"
                "After that, you will respond to Task 1 and Task 2 separately."
                "Click the mouse button (left or right) to start."
            ),
            color="black", height=0.7, wrapWidth=25, pos=(0, 0)
        )
        instruction_text.draw()
        self.win.flip()
        self.wait_for_mouse()

    #5. Show formal experiment instructions
    def formal_experiment_instructions(self):
        instruction_text = visual.TextStim(
            self.win,
            text=(
                "Next is the formal experiment.\n\n"
                "Task 1, Task 2, and Task 1+2 will appear in random order.\n"
                "There are a total of 96 trials.\n"
                "You will have a break after completing 48 trials.\n\n"
                "Click the left or right mouse button to start."
            ),
            color="black", height=0.7, wrapWidth=25, pos=(0, 0)
        )
        instruction_text.draw()
        self.win.flip()
        self.wait_for_mouse()

    #6. Break instructions
    def break_instructions(self):
        instructions = visual.TextStim(
            self.win,
            text=(
                "Have a Break.\n\n"
                "Click the mouse (left or right) to continue.\n\n"
            ),
            color="black", height=0.7, wrapWidth=25, pos=(0, 0)
        )
        instructions.draw()
        self.win.flip()

        while True:
            if any(self.mouse.getPressed()):
                break
            elif "escape" in event.getKeys():
                save_data_and_exit()
            core.wait(0.1)

    #7. End of experiment instructions
    def end_instructions(self):
        thank_you_text = visual.TextStim(
            self.win,
            text="Thank you for your participation!\n\n"
                 "The experiment will close automatically in 5 seconds.",
            color="black", height=0.7, wrapWidth=25, pos=(0, 0)
        )
        thank_you_text.draw()
        self.win.flip()
        core.wait(5.0)

# VWM stimulus presentation
def VWM_stimulation():
    global circle_radius, color_set, positions
    # Define circle positions
    angles_deg = [0, 60, 120, 180, 240, 300]  # Angles for six positions
    positions = [(stimulus_radius * math.cos(math.radians(angle)) * 1.1,  # Increase spacing by 10%
                  stimulus_radius * math.sin(math.radians(angle)) * 1.1) for angle in angles_deg]
    
    # Define ten colors
    color_set = ["lightblue", "darkgreen", "red", "grey", 
                 "lightgreen", "brown", "pink", "blue", "yellow", "purple"]
    
    # Define circle size (3.5 cm converted to degrees)
    circle_diameter = 3.5  # 3.5 cm corresponds to 3.5 degrees of visual angle
    circle_radius = circle_diameter / 2  # Convert to radius

def VWM_sample_display():
    sample_colors = random.sample(color_set, 3)  # Randomly pick three colors
    sample_positions = random.sample(positions, 3)  # Randomly pick three positions

    # Present stimuli for 400ms
    for i in range(3):
        visual.Circle(win, radius=circle_radius, fillColor=sample_colors[i], 
                      lineColor=sample_colors[i], pos=sample_positions[i]).draw()

    win.flip()
    core.wait(0.4)  # Show for 400ms

    return sample_colors, sample_positions

# MIT stimuli
def MIT_stimulation():
    global generate_safe_positions, flash_border, Tracking

    # Non-overlapping placement
    def generate_safe_positions(num_stimuli, image_size=(2.5, 2.5), min_dist=4.5):
        image_width, image_height = image_size
    
        # Automatically calculate boundaries
        screen_width_deg = win.size[0] / win.monitor.getSizePix()[0] * win.monitor.getWidth()
        screen_height_deg = screen_width_deg * (win.size[1] / win.size[0])  # 保持纵横比
    
        edge_margin = 0.2  # Unit: deg, buffer safety margin
        x_limit = screen_width_deg / 2 - image_width / 2 - edge_margin
        y_limit = screen_height_deg / 2 - image_height / 2 - edge_margin
        
        positions = []
        for _ in range(num_stimuli):
            while True:
                x = random.uniform(-x_limit, x_limit)
                y = random.uniform(-y_limit, y_limit)
                pos = np.array([x, y])
                if all(np.linalg.norm(pos - np.array(p)) > min_dist for p in positions):
                    positions.append((x, y))
                    break
        return positions
        
         # Border marking phase (show borders for 3 seconds without flashing)
    def flash_border(possible_n=[2, 3, 4, 5], duration=3.0):
        n = random.choice(possible_n)  # Randomly choose a count each time
        selected_images = random.sample(images, n)  # Randomly select n images
        target_MIT = []  # Used to record indices of selected images for later analysis
    
        for img in images:
            img.draw()
    
        for stimulus in selected_images:
            border = visual.Rect(
                win, 
                width=stimulus.size[0] + padding, 
                height=stimulus.size[1] + padding,
                pos=stimulus.pos, 
                lineColor='black', 
                fillColor=None, 
                lineWidth=4
            )
            border.draw()
            target_MIT.append(images.index(stimulus))
    
        win.flip()
        core.wait(duration)
        return target_MIT
        
        # Target tracking phase
    def Tracking(duration=7):# Duration Time
        speed = 6.3  # deg/sec
        image_width, image_height = images[0].size
    
        # Initialize random direction vectors
        angles = [random.uniform(0, 360) for _ in images]
        velocities = [[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in angles]
    
        # Screen boundaries
        screen_width_deg = win.size[0] / win.monitor.getSizePix()[0] * win.monitor.getWidth()
        screen_height_deg = screen_width_deg * (win.size[1] / win.size[0])
        x_limit = screen_width_deg / 2 - image_width / 2
        y_limit = screen_height_deg / 2 - image_height / 2
    
        timer = core.Clock()
        frame_clock = core.Clock()
    
        while timer.getTime() < duration:
            dt = frame_clock.getTime()
            frame_clock.reset()
    
            # Image movement
            for i, img in enumerate(images):
                dx = velocities[i][0] * speed * dt
                dy = velocities[i][1] * speed * dt
                new_x = img.pos[0] + dx
                new_y = img.pos[1] + dy
    
                bounced = False
                if abs(new_x) > x_limit:
                    velocities[i][0] *= -1
                    bounced = True
                if abs(new_y) > y_limit:
                    velocities[i][1] *= -1
                    bounced = True
    
                if bounced:
                    angle = np.degrees(np.arctan2(velocities[i][1], velocities[i][0]))
                    angle += random.uniform(-30, 30)
                    rad = np.radians(angle)
                    velocities[i][0] = np.cos(rad)
                    velocities[i][1] = np.sin(rad)
                    new_x = img.pos[0] + velocities[i][0] * speed * dt
                    new_y = img.pos[1] + velocities[i][1] * speed * dt
    
                img.pos = (new_x, new_y)
    
            # Avoid overlap
            min_dist = image_width
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    pos_i = np.array(images[i].pos)
                    pos_j = np.array(images[j].pos)
                    dist = np.linalg.norm(pos_i - pos_j)
    
                    if dist < min_dist:
                        for idx in [i, j]:
                            angle = np.degrees(np.arctan2(velocities[idx][1], velocities[idx][0]))
                            angle += random.uniform(150, 210)
                            rad = np.radians(angle)
                            velocities[idx][0] = np.cos(rad)
                            velocities[idx][1] = np.sin(rad)
    
                        images[i].pos = tuple(pos_i + np.array(velocities[i]) * speed * dt)
                        images[j].pos = tuple(pos_j + np.array(velocities[j]) * speed * dt)
    
            # Draw all images and refresh window each frame
            for img in images:
                img.draw()
            win.flip()

# VWM test phase (instructions and question presented simultaneously)
def VWM_test_phase_practice(sample_colors, sample_positions,):
    test_color = random.choice(sample_colors)

    if random.random() > 0.5:
        test_position = sample_positions[sample_colors.index(test_color)]
        condition = "Match"
    else:
        test_position = random.choice(
            [p for p in sample_positions if p != sample_positions[sample_colors.index(test_color)]]
        )
        condition = "Mismatch"

    # Create visual elements (practice colored circle + text prompt)
    test_circle = visual.Circle(win, radius=circle_radius, fillColor=test_color, lineColor=test_color, pos=test_position)
    question = visual.TextStim(win,
        text="Does the circle have the same color and position as before?\nLeft click = Yes, Right click = No",
        color="black",
        height=0.7,
        wrapWidth=25,
        pos=(0,0)
    )
    # Present simultaneously
    test_circle.draw()
    question.draw()
    win.flip()

    # Calculate reaction time and determine if answer is correct
    rt_clock = core.Clock()
    mouse = event.Mouse()
    rt_clock = core.Clock()
    while True:
        buttons = mouse.getPressed()
        if buttons[0]:  # Left mouse button = Yes
            rt = rt_clock.getTime()
            response = "Yes"
            key = "mouse_left"
            break
        elif buttons[2]:  # Right mouse button = No
            rt = rt_clock.getTime()
            response = "No"
            key = "mouse_right"
            break
        elif "escape" in event.getKeys():
            print("🚪 Participant exited early, saving data so far")
            save_data_and_exit()

    return test_color, test_position, condition, key, rt

# Test phase (formal experiment)
def VWM_test_phase(sample_colors, sample_positions):
    test_color = random.choice(sample_colors)

    if random.random() > 0.5:
        test_position = sample_positions[sample_colors.index(test_color)]
        condition = "Match"
    else:
        test_position = random.choice(
            [p for p in sample_positions if p != sample_positions[sample_colors.index(test_color)]]
        )
        condition = "Mismatch"

    test_circle = visual.Circle(win, radius=circle_radius, fillColor=test_color, lineColor=test_color, pos=test_position)
    question2 = visual.TextStim(win,
        text="Left = Yes, Right = No",
        color="black",
        height=0.7,
        wrapWidth=25,
        pos=(0,0)
    )
    # Present simultaneously
    test_circle.draw()
    question2.draw()
    win.flip()

    # Calculate reaction time and determine if answer is correct
    rt_clock = core.Clock()
    mouse = event.Mouse()
    rt_clock = core.Clock()
    while True:
        buttons = mouse.getPressed()
        if buttons[0]:  # Left mouse button = Yes
            rt = rt_clock.getTime()
            response = "Yes"
            key = "mouse_left"
            break
        elif buttons[2]:  # Right mouse button = No
            rt = rt_clock.getTime()
            response = "No"
            key = "mouse_right"
            break
        elif "escape" in event.getKeys():
            print("🚪 Participant exited early, saving data so far")
            save_data_and_exit()

    return test_color, test_position, condition, key, rt

# Image masking phase
# Simultaneous drawing (MIT single-task)
def masking(target_MIT, target_index, duration=3):
    # Randomly select test_index (target or distractor)
    test_index = target_index if random.random() < 0.5 else random.choice([i for i in range(len(images)) if i != target_index])

    # Draw all blockers
    for img in images:
        blocker = visual.Rect(
            win,
            width=img.size[0],
            height=img.size[1],
            pos=img.pos,
            lineColor='black',
            fillColor='black',
            lineWidth=2
        )
        blocker.draw()

    # Draw a mask border
    mask = visual.Rect(
        win,
        width=images[test_index].size[0] + 0.3,
        height=images[test_index].size[1] + 0.3,
        pos=images[test_index].pos,
        fillColor=None,
        lineColor='black',
        lineWidth=4
    )
    mask.draw()

    win.flip()
    core.wait(duration)

    return test_index  # Return for use in response phase

# Separate (for VWM+MIT dual task)
# Draw blockers separately
def masking_only_draw_blockers(win, images, target_index, duration=0.5):
    # Randomly select test_index (target or distractor)
    test_index = target_index if random.random() < 0.5 else random.choice(
        [i for i in range(len(images)) if i != target_index]
    )

    for img in images:
        blocker = visual.Rect(
            win,
            width=img.size[0],
            height=img.size[1],
            pos=img.pos,
            lineColor='black',
            fillColor='black',
            lineWidth=2
        )
        blocker.draw()
    win.flip()
    core.wait(duration)
    win.flip()
    return test_index  #Return index for drawing mask later

# Draw mask border separately
def masking_only_draw_mask_border(win, images, image, duration=2, padding=0.3):
    mask = visual.Rect(
        win,
        width=image.size[0] + padding,
        height=image.size[1] + padding,
        pos=image.pos,
        fillColor=None,
        lineColor='black',
        lineWidth=4
    )

    # Draw all blockers
    for img in images:
        blocker = visual.Rect(
            win,
            width=img.size[0],
            height=img.size[1],
            pos=img.pos,
            lineColor='black',
            fillColor='black',
            lineWidth=2
        )
        blocker.draw()

    mask.draw()
    win.flip()
    core.wait(duration)
    win.flip()

# MIT practice response
def MIT_test_practice(images, target_MIT, test_index, mode):
    is_target_masked = (test_index in target_MIT)
    condition = "Match" if is_target_masked else "Mismatch"

    # Set positions
    radius = 5
    angles = np.linspace(90, 90 + 360, 7, endpoint=False)
    positions = [(radius * np.cos(np.radians(a)), radius * np.sin(np.radians(a))) for a in angles]
    no_pos = positions[0]
    random.shuffle(positions[1:])
    for img, pos in zip(images, positions[1:]):
        img.pos = pos

    # Instruction prompt
    MIT_question_stim = visual.TextStim(
        win,
        text="If the target is in the border,\nclick the picture in the border.\nClick NO if the target is not in the border.",
        pos=(0, 0),
        height=0.4,
        color="black",
        wrapWidth=10
    )

    rt_timer = core.Clock()
    mouse = event.Mouse(win=win)
    mouse.setPos((0, 0))
    clicked = False
    result = "No Response"
    rt = None

    while not clicked:
        for img in images:
            img.draw()
        visual.TextStim(win, text="NO", pos=no_pos, height=1.5, color='black', bold=True).draw()
        MIT_question_stim.draw()
        win.flip()

        keys = event.getKeys()
        if 'escape' in keys:
            print("Experiment was terminated by user pressing ESC")
            save_data_and_exit()
        else:
            pass  # ignore invalid key presses

        if mouse.getPressed()[0]:
            temp_click_pos = mouse.getPos()  # record temporary click position
        
            # Check if click is within positions
            def inside_position(pos_list, click_pos, threshold=1.25):
                return any(np.linalg.norm(np.array(pos) - np.array(click_pos)) < threshold for pos in pos_list)
        
            if inside_position(positions, temp_click_pos):  # Only proceed if clicked within valid positions
                click_pos = temp_click_pos  # record valid click
                if condition == "Match":
                    result = "True" if np.linalg.norm(np.array(positions[test_index]) - np.array(click_pos)) < 1.25 else "Fault"
                else:
                    dx = no_pos[0] - click_pos[0]
                    dy = no_pos[1] - click_pos[1]
                    result = "True" if np.sqrt(dx**2 + dy**2) < 1.25 else "Fault"
        
                response_correct = (result == "True")
                rt = rt_timer.getTime()
                clicked = True  # Only exit loop on valid click

    return {
        "reaction_time": rt,
        "is_target_masked": is_target_masked,
        "mouse_click": click_pos,
        "condition": condition,
        "response": response_correct
    }

# MIT formal response
def MIT_test_phase(images, target_MIT, test_index, mode):
    is_target_masked = (test_index in target_MIT)
    condition = "Match" if is_target_masked else "Mismatch"

    radius = 5
    angles = np.linspace(90, 90 + 360, 7, endpoint=False)
    positions = [(radius * np.cos(np.radians(a)), radius * np.sin(np.radians(a))) for a in angles]
    no_pos = positions[0]
    random.shuffle(positions[1:])
    for img, pos in zip(images, positions[1:]):
        img.pos = pos

    rt_timer = core.Clock()
    mouse = event.Mouse(win=win)
    mouse.setPos((0, 0))
    clicked = False
    result = "No Response"
    rt = None

    while not clicked:
        for img in images:
            img.draw()
        visual.TextStim(win, text="NO", pos=no_pos, height=1.5, color='black', bold=True).draw()
        win.flip()

        if mouse.getPressed()[0]:
            temp_click_pos = mouse.getPos()  # record temporary click position
        
            # Check if click is within positions
            def inside_position(pos_list, click_pos, threshold=1.25):
                return any(np.linalg.norm(np.array(pos) - np.array(click_pos)) < threshold for pos in pos_list)
        
            if inside_position(positions, temp_click_pos):  # Only proceed if clicked within valid positions
                click_pos = temp_click_pos  # record valid click
                if condition == "Match":
                    result = "True" if np.linalg.norm(np.array(positions[test_index]) - np.array(click_pos)) < 1.25 else "Fault"
                else:
                    dx = no_pos[0] - click_pos[0]
                    dy = no_pos[1] - click_pos[1]
                    result = "True" if np.sqrt(dx**2 + dy**2) < 1.25 else "Fault"
        
                response_correct = (result == "True")
                rt = rt_timer.getTime()
                clicked = True  # Only exit loop on valid click

    return {
        "reaction_time": rt,
        "is_target_masked": is_target_masked,
        "mouse_click": click_pos,
        "condition": condition,
        "response": response_correct
    }

# Define MIT initial positions
def reset_images(images, size=(2.5, 2.5), min_dist=4.5):
    for img in images:
        img.size = size
    positions = generate_safe_positions(len(images), image_size=size, min_dist=min_dist)
    for i, img in enumerate(images):
        img.pos = positions[i]

# Ensure each possible n appears equally often
def generate_balanced_n_sequence(possible_n, total_trials):
    assert total_trials % len(possible_n) == 0, "总试次必须能被边框种类数整除"
    repeat_each = total_trials // len(possible_n)
    sequence = possible_n * repeat_each
    random.shuffle(sequence)
    return sequence

# Aggregation
data_list = [] # Global data collection list
# ==================== VWM trials ====================
def VWM_trial(is_practice, use_practice_phase=True, phase=""):
    Trial_begin("1")
    sample_colors, sample_positions = VWM_sample_display()
    Retention_Interval()
    if use_practice_phase:
        test_color, test_position, condition, key, rt = VWM_test_phase_practice(
            sample_colors, sample_positions
        )
    else:
        test_color, test_position, condition, key, rt = VWM_test_phase(
            sample_colors, sample_positions
        )
        
    # Determine correctness
    vwm_correct = (
        (condition == "Match" and key == "mouse_left") or 
        (condition == "Mismatch" and key == "mouse_right")
    )
    
    data_list.append({
        "phase": phase,
        "task": "VWM",
        "with_instruction": is_practice,
        "VWM_sample_colors": sample_colors,
        "VWM_sample_positions": sample_positions,  
        "VWM_test_color": test_color,
        "VWM_test_position": test_position,
        "VWM_condition": condition, # match or mismatch
        "VWM_key": key,
        "VWM_rt": rt,
        "VWM_correct": vwm_correct # whether participant answered correctly
    })
    core.wait(1.0)
    return vwm_correct

# ==================== MIT 试次 ====================
def MIT_trial(n, is_practice, phase=""):
    Trial_begin("2")
    reset_images(images)
    target_MIT = flash_border(possible_n=[n])
    target_index = random.choice(target_MIT)
    Tracking(duration=7) # tracking time
    test_index = masking(target_MIT, target_index)
    if is_practice:
        run_result = MIT_test_practice(images, target_MIT, test_index, mode="Practice_MIT")
    else:
        run_result = MIT_test_phase(images, target_MIT, test_index, mode="MIT")
    core.wait(1.0)
    
    mit_result = run_result
    target_filenames = [os.path.basename(images[idx].image) for idx in target_MIT]
    clicked_filename = os.path.basename(images[test_index].image) if mit_result["response"] else "NO"
    question_filename = os.path.basename(images[test_index].image)

    # determine if correct
    mit_correct = mit_result["response"]

    data_list.append({
    "phase": phase,
    "task": "MIT",
    "with_instruction": is_practice,
    "MIT_size": n,
    "MIT_target": target_filenames, # which images were tracked as targets
    "MIT_question": question_filename, # which image was highlighted after masking
    "MIT_condition": mit_result["condition"], # whether the highlighted image was a target (yes/no)
    "MIT_answer": clicked_filename, # which image the participant chose
    "MIT_correct": mit_result["response"], # whether the participant's answer was correct
    "MIT_rt": mit_result["reaction_time"] # participant's reaction time
    })

    return mit_result["response"]
    core.wait(1.0)

# ==================== VWM + MIT trials ====================
def VWM_MIT_trial(n, is_practice, phase=""):
    Trial_begin("1+2")
    sample_colors, sample_positions = VWM_sample_display() # VWM stimuli
    reset_images(images) # Initialize MIT images
    Fixation()
    target_MIT = flash_border(possible_n=[n]) # MIT borders
    target_index = random.choice(target_MIT) # select one as random target for test
    Tracking(duration=7) # tracking time
    test_index = masking_only_draw_blockers(win, images, target_index) # mask images
    test_image = images[test_index]   # store the image to be highlighted
    
    # === VWM response ===
    if is_practice:
        test_color, test_position, vwm_condition, key, rt = VWM_test_phase_practice(sample_colors, sample_positions)
    else:
        test_color, test_position, vwm_condition, key, rt = VWM_test_phase(sample_colors, sample_positions)

    # Check VWM response correctness
    vwm_correct = (
        (vwm_condition == "Match" and key == "mouse_left") or
        (vwm_condition == "Mismatch" and key == "mouse_right")
    )

    # === MIT response ===
    masking_only_draw_mask_border(win, images, test_image)
    if is_practice:
        mit_result = MIT_test_practice(images, target_MIT, test_index, mode="Practice_Combo")
    else:
        mit_result = MIT_test_phase(images, target_MIT, test_index, mode="Combo")

    # Check MIT response correctness
    mit_condition = mit_result["condition"]
    mit_key = mit_result["mouse_click"]
    target_filenames = [os.path.basename(images[idx].image) for idx in target_MIT]
    clicked_filename = os.path.basename(images[test_index].image) if mit_result["response"] else "NO"
    question_filename = os.path.basename(images[test_index].image)
    
    data_list.append({
        "phase": phase,
        "task": "VWM+MIT",
        "with_instruction": is_practice,

        # VWM part
        "VWM_sample_colors": sample_colors,
        "VWM_sample_positions": sample_positions,  
        "VWM_test_color": test_color,
        "VWM_test_position": test_position,
        "VWM_condition": vwm_condition, #match or mismatch
        "VWM_key": key,
        "VWM_rt": rt,
        "VWM_correct": vwm_correct, # whether participant answered correctly

        # MIT part
        "MIT_size": n,
        "MIT_target": target_filenames, #  which images were tracked as targets
        "MIT_question": question_filename,# which image was highlighted after masking
        "MIT_condition": mit_result["condition"],#whether the highlighted image was a target (yes/no)
        "MIT_answer": clicked_filename,# which image the participant chose
        "MIT_correct": mit_result["response"], # whether the participant's answer was correct
        "MIT_rt": mit_result["reaction_time"] # participant's reaction time
    })

    return vwm_correct, mit_result["response"]
    core.wait(1.0)

# ==================== Main control entry ====================
def practice_block(phase='practice'):
    instructions = Instructions(win)
    # VWM practice: with instructions + understanding confirmation + without instructions
    instructions.VWM_show_instructions()
    while True:
        for _ in range(2):
            VWM_trial(True, use_practice_phase=True, phase="practice")
            last = data_list[-1]
            show_feedback(vwm_correct=last["VWM_correct"])

        decision = confirm_understanding() # confirm understanding
        if decision == "continue":#  whether to repeat practice with instructions
            break
        elif decision == "repeat":
            continue  # practice again
    for _ in range(2):
        VWM_trial(True, use_practice_phase=False, phase="practice")
        last = data_list[-1]
        show_feedback(vwm_correct=last["VWM_correct"])
        core.wait(1.0)
    
    # MIT practice: 2 trials without prompt, then one with prompt for each n=2–5
    instructions.MIT_show_instructions()
    while True:
        for n in [2, 2]:
            MIT_trial(n=n, is_practice=True, phase="")
            show_feedback(mit_correct=data_list[-1]["MIT_correct"])
        if confirm_understanding() == "continue":
            break
    for n in [2, 3]:
        MIT_trial(n=n, is_practice=False, phase="")
        show_feedback(mit_correct=data_list[-1]["MIT_correct"])
        core.wait(1.0)

    # VWM+MIT practice: run one trial for each n=2–5
    instructions.VWM_MIT_show_instructions()
    while True:
        for n in [2, 2]:
            VWM_MIT_trial(n=n, is_practice=True, phase="")
            last = data_list[-1]
            show_feedback(vwm_correct=last["VWM_correct"], mit_correct=last["MIT_correct"])
        if confirm_understanding() == "continue":
            break
    for n in [2, 3, 4, 5]:
        VWM_MIT_trial(n=n, is_practice=False, phase="")
        last = data_list[-1]
        show_feedback(vwm_correct=last["VWM_correct"], mit_correct=last["MIT_correct"])

# Formal experiment block
def experiment_block(phase="formal"):
    global experiment_round
    instructions = Instructions(win)

    # Each block consists of 48 trials
    trial_units = []
    trial_units += [("VWM", None)] * 12 # 12 VWM trials
    for n in [2, 3, 4, 5]:
        trial_units += [("MIT", n)] * 4 #MIT: 4 trials for each n=2–5
    for n in [2, 3, 4, 5]:
        trial_units += [("VWM_MIT", n)] * 5 #V VWM+MIT: 5 trials for each n=2–5
    random.shuffle(trial_units)  # Randomize order

    for trial_type, n in trial_units:
        if trial_type == "VWM":
            VWM_trial(is_practice=False, use_practice_phase=False, phase="formal")
        elif trial_type == "MIT":
            MIT_trial(n=n, is_practice=False, phase="formal")
        elif trial_type == "VWM_MIT":
            VWM_MIT_trial(n=n, is_practice=False, phase="formal")
        core.wait(0.5)

# Main function structure
def main():
    global data_list
    VWM_stimulation()
    MIT_stimulation()

    instructions = Instructions(win)

    instructions.begin_instructions() # experiment start instructions
    practice_block(phase="practice") # practice
    instructions.formal_experiment_instructions() # formal experiment instructions
    print("✅ Formal experiment instructions displayed")
    experiment_block(phase="formal") # formal experiment
    instructions.end_instructions()
# Data saving (normal)
    print("✅ All trials completed")
    os.makedirs("data", exist_ok=True)
    filename = f"data/{exp_info['Participant']}"
    df = pd.DataFrame(data_list)  
    df.to_excel(filename + ".xlsx", index=False)
    df.to_csv(filename + ".csv", index=False)

# Data saving (exception protected)
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"💥 experiment breakdown：{e}")
        save_data_and_exit()
        raise
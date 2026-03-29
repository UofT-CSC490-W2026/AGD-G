import re

with open("agd/testing/test_edge.py", "r") as f:
    content = f.read()

content = content.replace('adv_img.save(filename)\n        print(f"Saved {filename}")', 'adv_img.save(filename)\n        print(f"Saved {filename}")\n        \n        padded_filename = f"padded_{safe_name}.png"\n        padded.save(padded_filename)\n        print(f"Saved {padded_filename}")')
content = content.replace('strength=0.0', 'strength=0.5')
content = content.replace('color="white"', 'color=(128, 128, 128)')

with open("agd/testing/test_edge.py", "w") as f:
    f.write(content)

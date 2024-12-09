from text2brick.models import LegoWorldData

def format_ldraw(data: LegoWorldData):
        #TODO update to support rotation matrix
        ldr_line = []
        for brick in data.world:
            ldr_line.append(f"1 {brick.brick_ref.color} {brick.x} {brick.y} {brick.z} 1 0 0 0 1 0 0 0 1 {brick.brick_ref.file_id}")
        return ldr_line

def save_ldr(data: LegoWorldData, filename: str):
    formatted_data = format_ldraw(data)
    with open(filename + ".ldr", "w") as file:
        for line in formatted_data:
            file.write(line + "\n")
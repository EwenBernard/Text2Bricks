from text2brick.models import LegoWorldData, BRICK_UNIT

def format_ldraw(data: LegoWorldData):
        #TODO update to support rotation matrix
        ldr_line = []
        for brick in data.world:
            ldr_line.append(f"1 {brick.brick_ref.color} {brick.x * BRICK_UNIT.W} {brick.y * BRICK_UNIT.H} {brick.z * BRICK_UNIT.D} 1 0 0 0 1 0 0 0 1 {brick.brick_ref.file_id}")
        return ldr_line

def save_ldr(data: LegoWorldData, filename: str):
    formatted_data = format_ldraw(data)
    with open(filename + ".ldr", "w") as file:
        for line in formatted_data:
            file.write(line + "\n")
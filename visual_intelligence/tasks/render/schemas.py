from pydantic import BaseModel


###
# Schemas
###
class RenderMetadata(BaseModel):
    grid_height: int
    grid_width: int
    ###
    image_height: int
    image_width: int

    grid_height_image: int
    grid_width_image: int
    grid_start_x_image: int
    grid_start_y_image: int


class RenderStyle(BaseModel):
    cell_size: int
    grid_border_size: int

    ###
    # Colors
    ###
    value_to_color: dict[int, tuple[int, int, int]]
    background_color: tuple[int, int, int]
    border_color: tuple[int, int, int]

    @property
    def color_to_value(self) -> dict[tuple[int, int, int], int]:
        return {v: k for k, v in self.value_to_color.items()}


###
# Defined based styles
###
ArcBaseStyle = RenderStyle(
    cell_size=30,
    grid_border_size=2,
    value_to_color={
        0: (0, 0, 0),  # Black
        1: (0, 116, 217),  # Blue
        2: (255, 65, 54),  # Red
        3: (46, 204, 64),  # Green
        4: (255, 220, 0),  # Yellow
        5: (170, 170, 170),  # Grey
        6: (240, 18, 190),  # Fuchsia
        7: (255, 133, 27),  # Orange
        8: (127, 219, 255),  # Teal
        9: (135, 12, 37),  # Brown
    },
    background_color=(0, 0, 0),  # Black background
    border_color=(85, 85, 85),  # Medium gray border
)

ArcExtendedStyle = RenderStyle(
    cell_size=30,
    grid_border_size=2,
    value_to_color={
        0: (0, 0, 0),  # Black
        1: (0, 116, 217),  # Blue
        2: (255, 65, 54),  # Red
        3: (46, 204, 64),  # Green
        4: (255, 220, 0),  # Yellow
        5: (170, 170, 170),  # Grey
        6: (240, 18, 190),  # Fuchsia
        7: (255, 133, 27),  # Orange
        8: (127, 219, 255),  # Teal
        9: (135, 12, 37),  # Brown
        10: (163, 73, 164),  # Purple (deep violet)
        11: (255, 182, 193),  # Pink (light pink)
        12: (0, 255, 255),  # Cyan (bright aqua)
        13: (128, 0, 128),  # Dark purple
        14: (192, 192, 192),  # Silver (light gray)
        15: (255, 255, 255),  # White
    },
    background_color=(0, 0, 0),  # Black background
    border_color=(85, 85, 85),  # Medium gray border
)


ChessStyle = RenderStyle(
    cell_size=30,
    grid_border_size=2,
    value_to_color={
        0: (0, 0, 0),  # Empty square background
        ### White pieces
        1: (144, 238, 144),  # White pawn (light green)
        2: (173, 216, 230),  # White knight (light blue)
        3: (216, 191, 216),  # White bishop (light purple)
        4: (176, 196, 222),  # White rook (light steel blue)
        5: (255, 182, 193),  # White queen (pink)
        6: (255, 215, 0),  # White king (gold)
        ### Black pieces
        7: (0, 100, 0),  # Black pawn (dark green)
        8: (0, 0, 139),  # Black knight (dark blue)
        9: (128, 0, 128),  # Black bishop (dark purple)
        10: (70, 130, 180),  # Black rook (steel blue)
        11: (139, 0, 0),  # Black queen (dark red)
        12: (184, 134, 11),  # Black king (dark goldenrod)
    },
    background_color=(0, 0, 0),
    border_color=(85, 85, 85),
)


MazeBaseStyle = RenderStyle(
    cell_size=16,
    grid_border_size=0,
    value_to_color={
        0: (71, 48, 45),  # Wall - dark brown
        1: (255, 255, 255),  # Path - white
        2: (244, 96, 54),  # End - orange-red
        3: (72, 191, 132),  # Start - green
        4: (46, 134, 171),  # Solution - blue
    },
    background_color=(255, 255, 255),  # White background
    border_color=(0, 0, 0),  # Black border
)

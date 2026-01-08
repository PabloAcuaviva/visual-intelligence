// ===========================================
// ARC Intermediate States Editor - Common Utilities
// ===========================================

// Grid class
class Grid {
    constructor(height, width, values) {
        this.height = parseInt(height, 10);
        this.width = parseInt(width, 10);
        
        if (isNaN(this.height) || this.height <= 0) {
            this.height = 1;
        }
        if (isNaN(this.width) || this.width <= 0) {
            this.width = 1;
        }

        this.grid = new Array(this.height);
        for (var i = 0; i < this.height; i++) {
            this.grid[i] = new Array(this.width);
            for (var j = 0; j < this.width; j++) {
                if (values !== undefined && values[i] !== undefined && values[i][j] !== undefined) {
                    this.grid[i][j] = values[i][j];
                } else {
                    this.grid[i][j] = 0;
                }
            }
        }
    }

    clone() {
        return new Grid(this.height, this.width, JSON.parse(JSON.stringify(this.grid)));
    }

    fill(value) {
        for (var i = 0; i < this.height; i++) {
            for (var j = 0; j < this.width; j++) {
                this.grid[i][j] = value;
            }
        }
    }
}

// Maximum cell size for display
var MAX_CELL_SIZE = 40;

// Flood fill algorithm
function floodfillFromLocation(grid, i, j, symbol) {
    i = parseInt(i);
    j = parseInt(j);
    symbol = parseInt(symbol);

    let target = grid[i][j];
    if (target == symbol) {
        return;
    }

    function flow(i, j, symbol, target) {
        if (i >= 0 && i < grid.length && j >= 0 && j < grid[i].length) {
            if (grid[i][j] == target) {
                grid[i][j] = symbol;
                flow(i - 1, j, symbol, target);
                flow(i + 1, j, symbol, target);
                flow(i, j - 1, symbol, target);
                flow(i, j + 1, symbol, target);
            }
        }
    }
    flow(i, j, symbol, target);
}

// Parse size tuple from string
function parseSizeTuple(size) {
    size = size.split('x');
    if (size.length != 2) {
        return null;
    }
    if ((size[0] < 1) || (size[1] < 1)) {
        return null;
    }
    if ((size[0] > 30) || (size[1] > 30)) {
        return null;
    }
    return size;
}

// Convert serialized grid array to Grid object
function convertSerializedGridToGridObject(values) {
    if (!values || values.length === 0) {
        return new Grid(1, 1);
    }
    let height = values.length;
    let width = values[0].length;
    return new Grid(height, width, values);
}

// Fit cells to container size
function fitCellsToContainer(jqGrid, height, width, containerHeight, containerWidth) {
    let candidate_height = Math.floor((containerHeight - height) / height);
    let candidate_width = Math.floor((containerWidth - width) / width);
    let size = Math.min(candidate_height, candidate_width);
    size = Math.min(MAX_CELL_SIZE, size);
    size = Math.max(8, size); // Minimum cell size
    jqGrid.find('.cell').css('height', size + 'px');
    jqGrid.find('.cell').css('width', size + 'px');
}

// Fill a jQuery grid element with data from a Grid object
function fillJqGridWithData(jqGrid, dataGrid) {
    jqGrid.empty();
    let height = dataGrid.height;
    let width = dataGrid.width;
    
    for (var i = 0; i < height; i++) {
        var row = $(document.createElement('div'));
        row.addClass('row');
        for (var j = 0; j < width; j++) {
            var cell = $(document.createElement('div'));
            cell.addClass('cell');
            cell.attr('x', i);
            cell.attr('y', j);
            setCellSymbol(cell, dataGrid.grid[i][j]);
            row.append(cell);
        }
        jqGrid.append(row);
    }
}

// Copy jQuery grid to data grid
function copyJqGridToDataGrid(jqGrid, dataGrid) {
    let row_count = jqGrid.find('.row').length;
    if (dataGrid.height != row_count) {
        return;
    }
    let col_count = jqGrid.find('.cell').length / row_count;
    if (dataGrid.width != col_count) {
        return;
    }
    jqGrid.find('.row').each(function(i, row) {
        $(row).find('.cell').each(function(j, cell) {
            dataGrid.grid[i][j] = parseInt($(cell).attr('symbol'));
        });
    });
}

// Set cell symbol with support for -1 (padding)
function setCellSymbol(cell, symbol) {
    symbol = parseInt(symbol);
    cell.attr('symbol', symbol);
    
    // Remove all symbol classes
    let classesToRemove = 'symbol_n1 ';
    for (let i = 0; i < 10; i++) {
        classesToRemove += 'symbol_' + i + ' ';
    }
    cell.removeClass(classesToRemove.trim());
    
    // Add appropriate class
    if (symbol === -1) {
        cell.addClass('symbol_n1');
    } else {
        cell.addClass('symbol_' + symbol);
    }
}

// Display error message
function errorMsg(msg) {
    $('#error_display').stop(true, true);
    $('#info_display').stop(true, true);

    $('#error_display').hide();
    $('#info_display').hide();
    $('#error_display').html(msg);
    $('#error_display').show();
    $('#error_display').fadeOut(5000);
}

// Display info message
function infoMsg(msg) {
    $('#error_display').stop(true, true);
    $('#info_display').stop(true, true);

    $('#info_display').hide();
    $('#error_display').hide();
    $('#info_display').html(msg);
    $('#info_display').show();
    $('#info_display').fadeOut(5000);
}

// ===========================================
// Grid Normalization Utilities
// ===========================================

// Normalize grid to target size, padding with -1
function normalizeGridToSize(grid, targetHeight, targetWidth) {
    let normalized = new Grid(targetHeight, targetWidth);
    normalized.fill(-1); // Fill with padding value
    
    // Copy original data
    for (let i = 0; i < Math.min(grid.height, targetHeight); i++) {
        for (let j = 0; j < Math.min(grid.width, targetWidth); j++) {
            normalized.grid[i][j] = grid.grid[i][j];
        }
    }
    
    return normalized;
}

// Get max dimensions from input and output grids
function getMaxDimensions(inputGrid, outputGrid) {
    let maxHeight = Math.max(inputGrid.height, outputGrid.height);
    let maxWidth = Math.max(inputGrid.width, outputGrid.width);
    return { height: maxHeight, width: maxWidth };
}

// Create empty grid with target size
function createEmptyGrid(height, width) {
    let grid = new Grid(height, width);
    grid.fill(0);
    return grid;
}

// ===========================================
// Canvas Rendering for Preview
// ===========================================

// Color map for symbols (including -1)
const SYMBOL_COLORS = {
    '-1': '#2d1f3d', // Padding - purple-ish dark (distinct from black)
    '0': '#000000',  // Black
    '1': '#0074D9',  // Blue
    '2': '#FF4136',  // Red
    '3': '#2ECC40',  // Green
    '4': '#FFDC00',  // Yellow
    '5': '#AAAAAA',  // Grey
    '6': '#F012BE',  // Fuchsia
    '7': '#FF851B',  // Orange
    '8': '#7FDBFF',  // Teal
    '9': '#870C25',  // Brown
};

// Render grid to canvas
function renderGridToCanvas(grid, canvas, maxSize = 80) {
    const ctx = canvas.getContext('2d');
    
    const cellSize = Math.min(
        Math.floor(maxSize / grid.height),
        Math.floor(maxSize / grid.width),
        15 // Max cell size for previews
    );
    
    canvas.width = grid.width * cellSize;
    canvas.height = grid.height * cellSize;
    
    for (let i = 0; i < grid.height; i++) {
        for (let j = 0; j < grid.width; j++) {
            const value = grid.grid[i][j];
            ctx.fillStyle = SYMBOL_COLORS[value.toString()] || SYMBOL_COLORS['0'];
            ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
            
            // Draw grid lines
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 0.5;
            ctx.strokeRect(j * cellSize, i * cellSize, cellSize, cellSize);
        }
    }
}

// ===========================================
// Copy/Paste Support
// ===========================================

var COPY_PASTE_DATA = [];

function copySelectedCells() {
    let selected = $('.ui-selected');
    if (selected.length === 0) {
        return false;
    }

    COPY_PASTE_DATA = [];
    for (let i = 0; i < selected.length; i++) {
        let x = parseInt($(selected[i]).attr('x'));
        let y = parseInt($(selected[i]).attr('y'));
        let symbol = parseInt($(selected[i]).attr('symbol'));
        COPY_PASTE_DATA.push([x, y, symbol]);
    }
    return true;
}

function pasteToLocation(jqGrid, targetX, targetY) {
    if (COPY_PASTE_DATA.length === 0) {
        return false;
    }

    let xs = COPY_PASTE_DATA.map(d => d[0]);
    let ys = COPY_PASTE_DATA.map(d => d[1]);
    let minX = Math.min(...xs);
    let minY = Math.min(...ys);

    for (let i = 0; i < COPY_PASTE_DATA.length; i++) {
        let x = COPY_PASTE_DATA[i][0];
        let y = COPY_PASTE_DATA[i][1];
        let symbol = COPY_PASTE_DATA[i][2];
        let newX = x - minX + targetX;
        let newY = y - minY + targetY;
        
        let res = jqGrid.find('[x="' + newX + '"][y="' + newY + '"]');
        if (res.length === 1) {
            setCellSymbol($(res[0]), symbol);
        }
    }
    return true;
}

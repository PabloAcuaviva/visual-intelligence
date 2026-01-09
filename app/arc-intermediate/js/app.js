// ===========================================
// ARC Intermediate States Editor - Main App
// ===========================================

// ===========================================
// State Management
// ===========================================

// All loaded tasks from folder
var LOADED_TASKS = [];
var CURRENT_TASK_INDEX = 0;

// Current task data
var CURRENT_TASK = null;  // { name, train: [...], test: [...] }

// Selected pair
var SELECTED_PAIR_TYPE = null;  // 'train' or 'test'
var SELECTED_PAIR_INDEX = 0;

// Timeline state
var CURRENT_STEP_INDEX = 0;  // 0 = input, 1..N = intermediate, N+1 = output
var NUM_INTERMEDIATE_STEPS = 8;  // Default number of intermediate steps

// Editor state
var CURRENT_GRID = null;
var IS_LOCKED = true;  // Input/Output locked by default

// Editor dimensions
var EDITOR_GRID_HEIGHT = 400;
var EDITOR_GRID_WIDTH = 400;

// Undo/Redo history
var UNDO_STACK = [];
var REDO_STACK = [];
var MAX_UNDO_HISTORY = 50;

// Line tool state
var LINE_START_CELL = null;  // {x, y} of first click for line tool

// Free draw state
var IS_DRAWING = false;  // Whether mouse is held down for free draw

// ===========================================
// Initialization
// ===========================================

$(document).ready(function() {
    // Set default steps
    $('#num_steps').val(NUM_INTERMEDIATE_STEPS);
    
    // Folder input handler
    $('#load_folder_input').on('change', handleFolderSelect);
    
    // Symbol picker
    $('#symbol_picker').find('.symbol_preview').click(function(event) {
        let symbol_preview = $(event.target);
        $('#symbol_picker').find('.symbol_preview').each(function(i, preview) {
            $(preview).removeClass('selected-symbol-preview');
        });
        symbol_preview.addClass('selected-symbol-preview');

        // If in select mode, fill selected cells
        let toolMode = $('input[name=tool_switching]:checked').val();
        if (toolMode === 'select') {
            let symbol = getSelectedSymbol();
            $('#editor_grid').find('.ui-selected').each(function(i, cell) {
                setCellSymbol($(cell), symbol);
            });
            syncFromEditorToData();
        }
    });

    // Tool switching
    $('input[type=radio][name=tool_switching]').change(function() {
        initializeSelectable();
        // Clear line start when switching tools
        LINE_START_CELL = null;
        IS_DRAWING = false;
        $('.cell').removeClass('line-start-marker');
        
        let toolMode = $('input[name=tool_switching]:checked').val();
        
        // Update cursor for draw mode
        if (toolMode === 'draw') {
            $('#editor_grid').addClass('draw-mode');
        } else {
            $('#editor_grid').removeClass('draw-mode');
        }
        
        if (toolMode === 'edit') {
            infoMsg('Edit mode: Click cells one at a time to change their color.');
        } else if (toolMode === 'draw') {
            infoMsg('Draw mode: Hold mouse and drag to paint cells continuously.');
        } else if (toolMode === 'select') {
            infoMsg('Select mode: Drag to select cells, then press C to copy or click a color to fill.');
        } else if (toolMode === 'line') {
            infoMsg('Line mode: Click start cell, then click end cell to draw a line.');
        } else if (toolMode === 'floodfill') {
            infoMsg('Flood fill mode: Click a cell to fill all connected cells of the same color.');
        }
    });

    // Keyboard shortcuts
    $(document).on('keydown', function(e) {
        // Arrow keys - context-dependent navigation
        if (e.key === 'ArrowLeft' && !e.ctrlKey && !e.metaKey) {
            e.preventDefault();
            // Check which modal is open
            if ($('#pair_preview_modal').is(':visible')) {
                prevPairPreview();
            } else if ($('#timeline_preview_modal').is(':visible')) {
                prevTimelinePreview();
            } else {
                prevStep();
            }
        } else if (e.key === 'ArrowRight' && !e.ctrlKey && !e.metaKey) {
            e.preventDefault();
            if ($('#pair_preview_modal').is(':visible')) {
                nextPairPreview();
            } else if ($('#timeline_preview_modal').is(':visible')) {
                nextTimelinePreview();
            } else {
                nextStep();
            }
        }
        
        // Ctrl+S / Cmd+S to save
        if ((e.ctrlKey || e.metaKey) && (e.key === 's' || e.key === 'S')) {
            e.preventDefault();
            saveCurrentTask();
        }
        
        // Undo: Ctrl+Z / Cmd+Z
        if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z') && !e.shiftKey) {
            e.preventDefault();
            undo();
        }
        
        // Redo: Ctrl+Y / Cmd+Shift+Z / Ctrl+Shift+Z
        if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || e.key === 'Y')) {
            e.preventDefault();
            redo();
        }
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === 'z' || e.key === 'Z')) {
            e.preventDefault();
            redo();
        }
        
        // Number keys for quick symbol select
        if (e.key >= '0' && e.key <= '9' && !e.ctrlKey && !e.metaKey) {
            let symbol = parseInt(e.key);
            selectSymbol(symbol);
        }
        
        // Copy: C key OR Ctrl+C / Cmd+C
        if (e.key === 'c' || e.key === 'C') {
            let selected = $('#editor_grid').find('.ui-selected');
            if (selected.length > 0) {
                e.preventDefault();  // Prevent default browser copy
                if (copySelectedCells()) {
                    infoMsg('Cells copied! Select a target cell and press V (or Ctrl+V) to paste.');
                }
            }
        }
        
        // Paste: V key OR Ctrl+V / Cmd+V
        if (e.key === 'v' || e.key === 'V') {
            e.preventDefault();  // Prevent default browser paste
            
            if (COPY_PASTE_DATA.length === 0) {
                errorMsg('No cells copied. Select cells and press C (or Ctrl+C) first.');
                return;
            }
            
            let selected = $('#editor_grid').find('.ui-selected');
            if (selected.length === 0) {
                errorMsg('Select a target cell on the grid first.');
                return;
            }
            
            if (selected.length === 1) {
                let x = parseInt(selected.attr('x'));
                let y = parseInt(selected.attr('y'));
                saveUndoState();
                if (pasteToLocation($('#editor_grid'), x, y)) {
                    syncFromEditorToData();
                    infoMsg('Pasted!');
                }
            } else {
                errorMsg('Select only ONE cell as paste destination.');
            }
        }
        
        // Escape - close modals or cancel line drawing
        if (e.key === 'Escape') {
            // Close pair preview modal if open
            if ($('#pair_preview_modal').is(':visible')) {
                closePairPreview();
                return;
            }
            // Close timeline preview modal if open
            if ($('#timeline_preview_modal').is(':visible')) {
                closeTimelinePreview();
                return;
            }
            // Close replace color modal if open
            if ($('#replace_color_modal').is(':visible')) {
                hideReplaceColorDialog();
                return;
            }
            // Cancel line drawing
            if (LINE_START_CELL) {
                LINE_START_CELL = null;
                $('.cell').removeClass('line-start-marker');
                infoMsg('Line drawing cancelled');
            }
        }
        
        // D for copy forward
        if ((e.key === 'd' || e.key === 'D') && !e.ctrlKey && !e.metaKey) {
            copyForward();
        }
        
        // ? for help toggle
        if (e.key === '?') {
            $('#shortcuts-help').toggle();
        }
    });
    
    // Click outside modal to close (pair preview modal)
    $('#pair_preview_modal').on('click', function(e) {
        // Only close if clicking the overlay itself, not the content
        if ($(e.target).is('#pair_preview_modal')) {
            closePairPreview();
        }
    });
    
    // Click outside modal to close (timeline preview modal)
    $('#timeline_preview_modal').on('click', function(e) {
        if ($(e.target).is('#timeline_preview_modal')) {
            closeTimelinePreview();
        }
    });

    // Initialize
    showEditorPlaceholder();
    updateUIState();
});

// ===========================================
// Folder Loading
// ===========================================

function handleFolderSelect(event) {
    let files = event.target.files;
    if (!files || files.length === 0) {
        errorMsg('No folder selected');
        return;
    }

    // Filter for JSON files
    let jsonFiles = [];
    for (let i = 0; i < files.length; i++) {
        if (files[i].name.endsWith('.json')) {
            jsonFiles.push(files[i]);
        }
    }

    if (jsonFiles.length === 0) {
        errorMsg('No JSON files found in folder');
        return;
    }

    // Sort by name
    jsonFiles.sort((a, b) => a.name.localeCompare(b.name));

    // Load all files
    LOADED_TASKS = [];
    let loadPromises = jsonFiles.map(file => {
        return new Promise((resolve, reject) => {
            let reader = new FileReader();
            reader.onload = function(e) {
                try {
                    let contents = JSON.parse(e.target.result);
                    resolve({
                        name: file.name,
                        data: contents
                    });
                } catch (err) {
                    console.warn('Failed to parse ' + file.name, err);
                    resolve(null);
                }
            };
            reader.onerror = () => resolve(null);
            reader.readAsText(file);
        });
    });

    Promise.all(loadPromises).then(results => {
        LOADED_TASKS = results.filter(r => r !== null);
        if (LOADED_TASKS.length === 0) {
            errorMsg('No valid JSON files loaded');
            return;
        }
        
        CURRENT_TASK_INDEX = 0;
        loadTask(0);
        infoMsg(`Loaded ${LOADED_TASKS.length} tasks from folder`);
    });
}

// ===========================================
// Task Navigation
// ===========================================

function loadTask(index) {
    if (index < 0 || index >= LOADED_TASKS.length) {
        return;
    }
    
    CURRENT_TASK_INDEX = index;
    let taskData = LOADED_TASKS[index];
    
    // Initialize task structure with intermediate grids
    CURRENT_TASK = {
        name: taskData.name,
        train: [],
        test: []
    };
    
    // Process train pairs
    if (taskData.data.train) {
        taskData.data.train.forEach((pair, i) => {
            let inputGrid = convertSerializedGridToGridObject(pair.input);
            let outputGrid = convertSerializedGridToGridObject(pair.output);
            let dims = getMaxDimensions(inputGrid, outputGrid);
            
            // Normalize grids to max size
            let normalizedInput = normalizeGridToSize(inputGrid, dims.height, dims.width);
            let normalizedOutput = normalizeGridToSize(outputGrid, dims.height, dims.width);
            
            // Load existing intermediate grids or create empty ones
            let intermediates = [];
            if (pair.intermediate_grids && pair.intermediate_grids.length > 0) {
                pair.intermediate_grids.forEach(ig => {
                    let grid = convertSerializedGridToGridObject(ig);
                    intermediates.push(normalizeGridToSize(grid, dims.height, dims.width));
                });
            } else {
                // Create empty intermediates
                for (let j = 0; j < NUM_INTERMEDIATE_STEPS; j++) {
                    intermediates.push(createEmptyGrid(dims.height, dims.width));
                }
            }
            
            CURRENT_TASK.train.push({
                input: normalizedInput,
                output: normalizedOutput,
                intermediates: intermediates,
                originalDims: {
                    inputHeight: inputGrid.height,
                    inputWidth: inputGrid.width,
                    outputHeight: outputGrid.height,
                    outputWidth: outputGrid.width
                }
            });
        });
    }
    
    // Process test pairs
    if (taskData.data.test) {
        taskData.data.test.forEach((pair, i) => {
            let inputGrid = convertSerializedGridToGridObject(pair.input);
            let outputGrid = convertSerializedGridToGridObject(pair.output);
            let dims = getMaxDimensions(inputGrid, outputGrid);
            
            let normalizedInput = normalizeGridToSize(inputGrid, dims.height, dims.width);
            let normalizedOutput = normalizeGridToSize(outputGrid, dims.height, dims.width);
            
            let intermediates = [];
            if (pair.intermediate_grids && pair.intermediate_grids.length > 0) {
                pair.intermediate_grids.forEach(ig => {
                    let grid = convertSerializedGridToGridObject(ig);
                    intermediates.push(normalizeGridToSize(grid, dims.height, dims.width));
                });
            } else {
                for (let j = 0; j < NUM_INTERMEDIATE_STEPS; j++) {
                    intermediates.push(createEmptyGrid(dims.height, dims.width));
                }
            }
            
            CURRENT_TASK.test.push({
                input: normalizedInput,
                output: normalizedOutput,
                intermediates: intermediates,
                originalDims: {
                    inputHeight: inputGrid.height,
                    inputWidth: inputGrid.width,
                    outputHeight: outputGrid.height,
                    outputWidth: outputGrid.width
                }
            });
        });
    }
    
    // IMPORTANT: Clear state BEFORE selecting pair to avoid syncing old data to new task
    CURRENT_GRID = null;
    HISTORY_STACK = [];
    HISTORY_POINTER = -1;
    SELECTED_PAIR_TYPE = null;
    SELECTED_PAIR_INDEX = null;
    CURRENT_STEP_INDEX = 0;
    
    // Update UI
    updateTaskDisplay();
    renderPairsList();
    
    // Select first train pair by default (will start at input, step 0)
    if (CURRENT_TASK.train.length > 0) {
        selectPairFresh('train', 0);
    } else if (CURRENT_TASK.test.length > 0) {
        selectPairFresh('test', 0);
    }
    
    updateUIState();
}

function prevTask() {
    if (CURRENT_TASK_INDEX > 0) {
        loadTask(CURRENT_TASK_INDEX - 1);
    }
}

function nextTask() {
    if (CURRENT_TASK_INDEX < LOADED_TASKS.length - 1) {
        loadTask(CURRENT_TASK_INDEX + 1);
    }
}

function updateTaskDisplay() {
    if (CURRENT_TASK) {
        $('#task_name').text(CURRENT_TASK.name);
        $('#task_counter').text(`Task ${CURRENT_TASK_INDEX + 1} of ${LOADED_TASKS.length}`);
    } else {
        $('#task_name').text('No task loaded');
        $('#task_counter').text('');
    }
}

function updateUIState() {
    // Task navigation buttons
    $('#prev_task_btn').prop('disabled', CURRENT_TASK_INDEX <= 0);
    $('#next_task_btn').prop('disabled', CURRENT_TASK_INDEX >= LOADED_TASKS.length - 1);
    
    // Delete step button (can't delete input/output)
    let canDelete = CURRENT_STEP_INDEX > 0 && 
                    CURRENT_STEP_INDEX <= getCurrentPair()?.intermediates.length;
    $('#delete_step_btn2').prop('disabled', !canDelete);
    
    // Update lock button text
    $('#lock_btn').html(IS_LOCKED ? '🔒 Locked' : '🔓 Unlocked');
}

// ===========================================
// Pairs List
// ===========================================

function renderPairsList() {
    if (!CURRENT_TASK) return;
    
    // Render train pairs
    let trainContainer = $('#train_pairs_list');
    trainContainer.empty();
    CURRENT_TASK.train.forEach((pair, index) => {
        trainContainer.append(createPairItem('train', index, pair));
    });
    
    // Render test pairs
    let testContainer = $('#test_pairs_list');
    testContainer.empty();
    CURRENT_TASK.test.forEach((pair, index) => {
        testContainer.append(createPairItem('test', index, pair));
    });
}

function createPairItem(type, index, pair) {
    let hasIntermediates = pair.intermediates.some(g => !isGridEmpty(g));
    let stepCount = pair.intermediates.length;
    
    let item = $(`
        <div class="pair-item" data-type="${type}" data-index="${index}">
            <button class="pair-zoom-btn" title="View larger (fullscreen)">🔍</button>
            <div class="pair-item-header">
                <span class="pair-item-title">${type === 'train' ? 'Train' : 'Test'} ${index + 1}</span>
                <span class="pair-item-badge ${hasIntermediates ? '' : 'empty'}">
                    ${hasIntermediates ? `✓ ${stepCount} steps` : 'empty'}
                </span>
            </div>
            <div class="pair-grids">
                <div class="pair-grid-preview" id="${type}_input_preview_${index}"></div>
                <span class="pair-arrow">→</span>
                <div class="pair-grid-preview" id="${type}_output_preview_${index}"></div>
            </div>
        </div>
    `);
    
    // Click handler for selection
    item.on('click', function(e) {
        // Don't select if clicking zoom button
        if ($(e.target).hasClass('pair-zoom-btn')) return;
        selectPair(type, index);
    });
    
    // Zoom button handler
    item.find('.pair-zoom-btn').on('click', function(e) {
        e.stopPropagation();
        openPairPreview(type, index);
    });
    
    // Render preview canvases after adding to DOM
    setTimeout(() => {
        let inputCanvas = document.createElement('canvas');
        let outputCanvas = document.createElement('canvas');
        renderGridToCanvas(pair.input, inputCanvas, 60);
        renderGridToCanvas(pair.output, outputCanvas, 60);
        $(`#${type}_input_preview_${index}`).append(inputCanvas);
        $(`#${type}_output_preview_${index}`).append(outputCanvas);
    }, 0);
    
    return item;
}

function isGridEmpty(grid) {
    for (let i = 0; i < grid.height; i++) {
        for (let j = 0; j < grid.width; j++) {
            if (grid.grid[i][j] !== 0) {
                return false;
            }
        }
    }
    return true;
}

function selectPair(type, index) {
    // Save current edits first (when switching pairs within same task)
    syncFromEditorToData();
    
    selectPairFresh(type, index);
}

// Used when loading a new task - skips syncing old data
function selectPairFresh(type, index) {
    SELECTED_PAIR_TYPE = type;
    SELECTED_PAIR_INDEX = index;
    CURRENT_STEP_INDEX = 0;  // Start at input
    
    // Clear history for fresh start
    HISTORY_STACK = [];
    HISTORY_POINTER = -1;
    
    // Update visual selection
    $('.pair-item').removeClass('active');
    $(`.pair-item[data-type="${type}"][data-index="${index}"]`).addClass('active');
    
    // Render timeline
    renderTimeline();
    
    // Load first step (input)
    loadStepIntoEditor(0);
    
    updateUIState();
}

function getCurrentPair() {
    if (!CURRENT_TASK || !SELECTED_PAIR_TYPE) return null;
    
    let pairs = SELECTED_PAIR_TYPE === 'train' ? CURRENT_TASK.train : CURRENT_TASK.test;
    return pairs[SELECTED_PAIR_INDEX] || null;
}

// ===========================================
// Timeline
// ===========================================

function renderTimeline() {
    let pair = getCurrentPair();
    if (!pair) {
        $('#timeline').html('<div class="timeline-placeholder">Select a pair to view timeline</div>');
        return;
    }
    
    let timeline = $('#timeline');
    timeline.empty();
    
    // Input step
    let inputStep = createTimelineStep('INPUT', pair.input, 0, true);
    timeline.append(inputStep);
    
    // Arrow
    timeline.append('<div class="timeline-arrow">→</div>');
    
    // Intermediate steps
    pair.intermediates.forEach((grid, i) => {
        let stepNum = i + 1;
        let step = createTimelineStep(`Step ${stepNum}`, grid, stepNum, false);
        timeline.append(step);
        timeline.append('<div class="timeline-arrow">→</div>');
    });
    
    // Output step
    let outputIndex = pair.intermediates.length + 1;
    let outputStep = createTimelineStep('OUTPUT', pair.output, outputIndex, true);
    timeline.append(outputStep);
    
    // Highlight current step
    updateTimelineSelection();
    updateStepCounter();
}

function createTimelineStep(label, grid, index, isLocked) {
    let step = $(`
        <div class="timeline-step ${isLocked ? 'locked' : ''}" data-step="${index}">
            <button class="step-zoom-btn" title="View larger">🔍</button>
            <div class="step-preview"></div>
            <span class="step-label">${label}${isLocked ? ' <span class="lock-icon">🔒</span>' : ''}</span>
        </div>
    `);
    
    // Render canvas
    let canvas = document.createElement('canvas');
    renderGridToCanvas(grid, canvas, 80);
    step.find('.step-preview').append(canvas);
    
    // Click handler for selection
    step.on('click', function(e) {
        // Don't select if clicking zoom button
        if ($(e.target).hasClass('step-zoom-btn')) return;
        let stepIndex = parseInt($(this).attr('data-step'));
        loadStepIntoEditor(stepIndex);
    });
    
    // Zoom button handler
    step.find('.step-zoom-btn').on('click', function(e) {
        e.stopPropagation();
        openTimelinePreview(index);
    });
    
    return step;
}

function updateTimelineSelection() {
    $('.timeline-step').removeClass('active');
    $(`.timeline-step[data-step="${CURRENT_STEP_INDEX}"]`).addClass('active');
}

function updateStepCounter() {
    let pair = getCurrentPair();
    if (!pair) {
        $('#step_counter').text('Step 0 / 0');
        return;
    }
    
    let total = pair.intermediates.length + 2;  // input + intermediates + output
    $('#step_counter').text(`Step ${CURRENT_STEP_INDEX + 1} / ${total}`);
}

// ===========================================
// Step Navigation
// ===========================================

function prevStep() {
    if (CURRENT_STEP_INDEX > 0) {
        syncFromEditorToData();
        loadStepIntoEditor(CURRENT_STEP_INDEX - 1);
    }
}

function nextStep() {
    let pair = getCurrentPair();
    if (!pair) return;
    
    let maxIndex = pair.intermediates.length + 1;  // +1 for output
    if (CURRENT_STEP_INDEX < maxIndex) {
        syncFromEditorToData();
        loadStepIntoEditor(CURRENT_STEP_INDEX + 1);
    }
}

function loadStepIntoEditor(stepIndex) {
    let pair = getCurrentPair();
    if (!pair) return;
    
    CURRENT_STEP_INDEX = stepIndex;
    
    // Determine which grid to load
    let grid;
    let isInputOrOutput = false;
    
    if (stepIndex === 0) {
        grid = pair.input;
        isInputOrOutput = true;
        $('#editing_status').text('Editing: INPUT (locked by default)');
    } else if (stepIndex === pair.intermediates.length + 1) {
        grid = pair.output;
        isInputOrOutput = true;
        $('#editing_status').text('Editing: OUTPUT (locked by default)');
    } else {
        grid = pair.intermediates[stepIndex - 1];
        isInputOrOutput = false;
        $('#editing_status').text(`Editing: Step ${stepIndex} of ${pair.intermediates.length}`);
    }
    
    CURRENT_GRID = grid.clone();
    
    // Render to editor
    refreshEditorGrid();
    
    // Update grid size display
    $('#grid_size_display').val(`${grid.height}x${grid.width}`);
    
    // Update lock state for input/output
    if (isInputOrOutput) {
        IS_LOCKED = true;
    }
    
    updateTimelineSelection();
    updateStepCounter();
    updateUIState();
}

// ===========================================
// Editor
// ===========================================

function refreshEditorGrid() {
    if (!CURRENT_GRID) {
        showEditorPlaceholder();
        return;
    }
    
    let jqGrid = $('#editor_grid');
    jqGrid.empty();
    jqGrid.removeClass('editor_placeholder');
    
    fillJqGridWithData(jqGrid, CURRENT_GRID);
    setUpEditorListeners(jqGrid);
    fitCellsToContainer(jqGrid, CURRENT_GRID.height, CURRENT_GRID.width, EDITOR_GRID_HEIGHT, EDITOR_GRID_WIDTH);
    initializeSelectable();
}

function showEditorPlaceholder() {
    let jqGrid = $('#editor_grid');
    jqGrid.empty();
    jqGrid.append('<div class="editor_placeholder">Load a task folder and select a pair to begin editing.</div>');
}

function setUpEditorListeners(jqGrid) {
    // Click handler for edit, floodfill, line tools
    jqGrid.find('.cell').on('click', function(event) {
        let cell = $(event.target);
        let mode = $('input[name=tool_switching]:checked').val();
        
        // Line tool has special handling
        if (mode === 'line') {
            handleLineToolClick(cell);
            return;
        }
        
        // Check if locked for other tools
        if (isCurrentStepLocked()) {
            errorMsg('This step is locked. Click "Unlock" to edit.');
            return;
        }
        
        let symbol = getSelectedSymbol();

        if (mode === 'floodfill') {
            saveUndoState();
            floodfillFromLocation(CURRENT_GRID.grid, cell.attr('x'), cell.attr('y'), symbol);
            refreshEditorGrid();
            syncFromEditorToData();
        } else if (mode === 'edit') {
            saveUndoState();
            setCellSymbol(cell, symbol);
            syncFromEditorToData();
        }
    });
    
    // Free draw: mousedown to start drawing
    jqGrid.find('.cell').on('mousedown', function(event) {
        let mode = $('input[name=tool_switching]:checked').val();
        if (mode !== 'draw') return;
        
        if (isCurrentStepLocked()) {
            errorMsg('This step is locked. Click "Unlock" to edit.');
            return;
        }
        
        event.preventDefault();
        IS_DRAWING = true;
        saveUndoState();
        
        let cell = $(event.target);
        let symbol = getSelectedSymbol();
        setCellSymbol(cell, symbol);
        
        // Update grid data
        let x = parseInt(cell.attr('x'));
        let y = parseInt(cell.attr('y'));
        CURRENT_GRID.grid[x][y] = symbol;
    });
    
    // Free draw: mousemove while drawing
    jqGrid.find('.cell').on('mouseenter', function(event) {
        let mode = $('input[name=tool_switching]:checked').val();
        if (mode !== 'draw' || !IS_DRAWING) return;
        
        let cell = $(event.target);
        let symbol = getSelectedSymbol();
        setCellSymbol(cell, symbol);
        
        // Update grid data
        let x = parseInt(cell.attr('x'));
        let y = parseInt(cell.attr('y'));
        CURRENT_GRID.grid[x][y] = symbol;
    });
    
    // Free draw: mouseup to stop drawing
    $(document).on('mouseup.freedraw', function() {
        if (IS_DRAWING) {
            IS_DRAWING = false;
            syncFromEditorToData();
        }
    });
}

function getSelectedSymbol() {
    let selected = $('#symbol_picker .selected-symbol-preview')[0];
    return parseInt($(selected).attr('symbol'));
}

function selectSymbol(symbol) {
    $('#symbol_picker').find('.symbol_preview').removeClass('selected-symbol-preview');
    $(`.symbol_preview[symbol="${symbol}"]`).addClass('selected-symbol-preview');
}

function initializeSelectable() {
    let editorGrid = $('#editor_grid');
    try {
        if (editorGrid.hasClass('ui-selectable')) {
            editorGrid.selectable('destroy');
        }
    } catch (e) {
        // Ignore
    }
    
    let toolMode = $('input[name=tool_switching]:checked').val();
    if (toolMode === 'select') {
        editorGrid.selectable({
            autoRefresh: false,
            filter: '> .row > .cell',
            start: function(event, ui) {
                editorGrid.find('.ui-selected').removeClass('ui-selected');
            }
        });
    }
}

function isCurrentStepLocked() {
    let pair = getCurrentPair();
    if (!pair) return true;
    
    // Input or output
    let isInputOrOutput = CURRENT_STEP_INDEX === 0 || 
                          CURRENT_STEP_INDEX === pair.intermediates.length + 1;
    
    return isInputOrOutput && IS_LOCKED;
}

function toggleLock() {
    IS_LOCKED = !IS_LOCKED;
    updateUIState();
    
    if (IS_LOCKED) {
        infoMsg('Input/Output grids are now LOCKED');
    } else {
        infoMsg('Input/Output grids are now UNLOCKED - be careful!');
    }
}

// ===========================================
// Data Synchronization
// ===========================================

function syncFromEditorToData() {
    if (!CURRENT_GRID) return;
    
    let pair = getCurrentPair();
    if (!pair) return;
    
    // Copy from editor to CURRENT_GRID
    copyJqGridToDataGrid($('#editor_grid'), CURRENT_GRID);
    
    // Update the appropriate grid in the pair
    if (CURRENT_STEP_INDEX === 0) {
        pair.input = CURRENT_GRID.clone();
    } else if (CURRENT_STEP_INDEX === pair.intermediates.length + 1) {
        pair.output = CURRENT_GRID.clone();
    } else {
        pair.intermediates[CURRENT_STEP_INDEX - 1] = CURRENT_GRID.clone();
    }
    
    // Update timeline preview
    updateTimelineStepPreview(CURRENT_STEP_INDEX);
    
    // Update pair badge in list
    updatePairBadge();
}

function updateTimelineStepPreview(stepIndex) {
    let step = $(`.timeline-step[data-step="${stepIndex}"]`);
    if (step.length === 0) return;
    
    let canvas = step.find('canvas')[0];
    if (canvas && CURRENT_GRID) {
        renderGridToCanvas(CURRENT_GRID, canvas, 80);
    }
}

function updatePairBadge() {
    let pair = getCurrentPair();
    if (!pair) return;
    
    let hasIntermediates = pair.intermediates.some(g => !isGridEmpty(g));
    let badge = $(`.pair-item[data-type="${SELECTED_PAIR_TYPE}"][data-index="${SELECTED_PAIR_INDEX}"]`)
                .find('.pair-item-badge');
    
    if (hasIntermediates) {
        badge.removeClass('empty');
        badge.text(`✓ ${pair.intermediates.length} steps`);
    } else {
        badge.addClass('empty');
        badge.text('empty');
    }
}

// ===========================================
// Step Management
// ===========================================

function addStepBefore() {
    let pair = getCurrentPair();
    if (!pair) return;
    
    // Can only add before intermediate steps or output
    if (CURRENT_STEP_INDEX === 0) {
        errorMsg('Cannot add step before INPUT');
        return;
    }
    
    syncFromEditorToData();
    
    // Create empty grid
    let newGrid = createEmptyGrid(pair.input.height, pair.input.width);
    
    // Insert into intermediates
    let insertIndex = CURRENT_STEP_INDEX - 1;
    pair.intermediates.splice(insertIndex, 0, newGrid);
    
    renderTimeline();
    loadStepIntoEditor(CURRENT_STEP_INDEX);  // Stay at same position (now it's the new step)
    
    infoMsg('Added new step before current');
}

function addStepAfter() {
    let pair = getCurrentPair();
    if (!pair) return;
    
    // Can only add after input or intermediate steps
    if (CURRENT_STEP_INDEX === pair.intermediates.length + 1) {
        errorMsg('Cannot add step after OUTPUT');
        return;
    }
    
    syncFromEditorToData();
    
    // Create empty grid
    let newGrid = createEmptyGrid(pair.input.height, pair.input.width);
    
    // Insert into intermediates
    let insertIndex = CURRENT_STEP_INDEX;  // After current step in intermediates array
    pair.intermediates.splice(insertIndex, 0, newGrid);
    
    renderTimeline();
    loadStepIntoEditor(CURRENT_STEP_INDEX + 1);  // Move to new step
    
    infoMsg('Added new step after current');
}

function deleteCurrentStep() {
    let pair = getCurrentPair();
    if (!pair) return;
    
    // Can only delete intermediate steps
    if (CURRENT_STEP_INDEX === 0) {
        errorMsg('Cannot delete INPUT');
        return;
    }
    if (CURRENT_STEP_INDEX === pair.intermediates.length + 1) {
        errorMsg('Cannot delete OUTPUT');
        return;
    }
    
    // Remove from intermediates
    let removeIndex = CURRENT_STEP_INDEX - 1;
    pair.intermediates.splice(removeIndex, 1);
    
    renderTimeline();
    
    // Adjust current step if needed
    let newIndex = Math.min(CURRENT_STEP_INDEX, pair.intermediates.length);
    loadStepIntoEditor(newIndex);
    
    infoMsg('Step deleted');
}

function applyStepCount() {
    let newCount = parseInt($('#num_steps').val());
    if (isNaN(newCount) || newCount < 1 || newCount > 50) {
        errorMsg('Step count must be between 1 and 50');
        return;
    }
    
    NUM_INTERMEDIATE_STEPS = newCount;
    
    // Apply to current pair if loaded
    let pair = getCurrentPair();
    if (pair) {
        let currentCount = pair.intermediates.length;
        
        if (newCount > currentCount) {
            // Add empty steps
            for (let i = currentCount; i < newCount; i++) {
                pair.intermediates.push(createEmptyGrid(pair.input.height, pair.input.width));
            }
        } else if (newCount < currentCount) {
            // Remove steps from end
            pair.intermediates.splice(newCount);
        }
        
        renderTimeline();
        updatePairBadge();
    }
    
    infoMsg(`Intermediate steps set to ${newCount}`);
}

// ===========================================
// Editor Actions
// ===========================================

function copyFromPrevious() {
    let pair = getCurrentPair();
    if (!pair) return;
    
    if (CURRENT_STEP_INDEX === 0) {
        errorMsg('No previous step to copy from');
        return;
    }
    
    if (isCurrentStepLocked()) {
        errorMsg('Unlock this step first to copy');
        return;
    }
    
    // Get previous grid
    let prevGrid;
    if (CURRENT_STEP_INDEX === 1) {
        prevGrid = pair.input;
    } else if (CURRENT_STEP_INDEX === pair.intermediates.length + 1) {
        prevGrid = pair.intermediates[pair.intermediates.length - 1];
    } else {
        prevGrid = pair.intermediates[CURRENT_STEP_INDEX - 2];
    }
    
    CURRENT_GRID = prevGrid.clone();
    refreshEditorGrid();
    syncFromEditorToData();
    
    infoMsg('Copied from previous step');
}

function resetCurrentGrid() {
    if (!CURRENT_GRID) return;
    
    if (isCurrentStepLocked()) {
        errorMsg('Unlock this step first to reset');
        return;
    }
    
    CURRENT_GRID.fill(0);
    refreshEditorGrid();
    syncFromEditorToData();
    
    infoMsg('Grid reset to empty');
}

// ===========================================
// Save Task (with File System Access API)
// ===========================================

// Store the directory handle for saving multiple files
var SAVE_DIRECTORY_HANDLE = null;

async function pickSaveFolder() {
    try {
        SAVE_DIRECTORY_HANDLE = await window.showDirectoryPicker({
            mode: 'readwrite'
        });
        $('#save_folder_path').val(SAVE_DIRECTORY_HANDLE.name);
        infoMsg(`Save folder set: ${SAVE_DIRECTORY_HANDLE.name}`);
        return true;
    } catch (err) {
        if (err.name !== 'AbortError') {
            errorMsg('Failed to select folder: ' + err.message);
        }
        return false;
    }
}

async function saveCurrentTask() {
    if (!CURRENT_TASK) {
        errorMsg('No task loaded to save');
        return;
    }
    
    // If no folder selected, prompt user to pick one
    if (!SAVE_DIRECTORY_HANDLE) {
        let picked = await pickSaveFolder();
        if (!picked) return;
    }
    
    // Sync current edits
    syncFromEditorToData();
    
    // Prepare output JSON
    let output = {
        train: [],
        test: []
    };
    
    // Process train pairs
    CURRENT_TASK.train.forEach((pair, i) => {
        let trainPair = {
            input: denormalizeGrid(pair.input, pair.originalDims.inputHeight, pair.originalDims.inputWidth),
            output: denormalizeGrid(pair.output, pair.originalDims.outputHeight, pair.originalDims.outputWidth),
        };
        
        // Only include intermediate_grids if there are non-empty ones
        let hasContent = pair.intermediates.some(g => !isGridEmpty(g));
        if (hasContent) {
            trainPair.intermediate_grids = pair.intermediates.map(g => g.grid);
        }
        
        output.train.push(trainPair);
    });
    
    // Process test pairs
    CURRENT_TASK.test.forEach((pair, i) => {
        let testPair = {
            input: denormalizeGrid(pair.input, pair.originalDims.inputHeight, pair.originalDims.inputWidth),
            output: denormalizeGrid(pair.output, pair.originalDims.outputHeight, pair.originalDims.outputWidth),
        };
        
        let hasContent = pair.intermediates.some(g => !isGridEmpty(g));
        if (hasContent) {
            testPair.intermediate_grids = pair.intermediates.map(g => g.grid);
        }
        
        output.test.push(testPair);
    });
    
    // Save to selected folder using File System Access API
    let filename = CURRENT_TASK.name;
    let jsonStr = JSON.stringify(output, null, 2);
    
    try {
        const fileHandle = await SAVE_DIRECTORY_HANDLE.getFileHandle(filename, { create: true });
        const writable = await fileHandle.createWritable();
        await writable.write(jsonStr);
        await writable.close();
        infoMsg(`Saved: ${SAVE_DIRECTORY_HANDLE.name}/${filename}`);
    } catch (err) {
        errorMsg('Failed to save: ' + err.message);
        // Reset handle if permission was revoked
        if (err.name === 'NotAllowedError') {
            SAVE_DIRECTORY_HANDLE = null;
            $('#save_folder_path').val('');
        }
    }
}

// Denormalize grid - remove padding (-1) and restore original size
function denormalizeGrid(grid, originalHeight, originalWidth) {
    let result = [];
    for (let i = 0; i < originalHeight; i++) {
        let row = [];
        for (let j = 0; j < originalWidth; j++) {
            let val = grid.grid[i][j];
            // Convert -1 back to 0 if it was padding
            row.push(val === -1 ? 0 : val);
        }
        result.push(row);
    }
    return result;
}

// ===========================================
// Undo/Redo System
// ===========================================

function saveUndoState() {
    if (!CURRENT_GRID) return;
    
    // Save current state to undo stack
    UNDO_STACK.push(CURRENT_GRID.clone());
    
    // Limit stack size
    if (UNDO_STACK.length > MAX_UNDO_HISTORY) {
        UNDO_STACK.shift();
    }
    
    // Clear redo stack when new action is taken
    REDO_STACK = [];
    
    updateUndoRedoButtons();
}

function undo() {
    if (UNDO_STACK.length === 0) {
        errorMsg('Nothing to undo');
        return;
    }
    
    if (!CURRENT_GRID) return;
    
    // Save current state to redo stack
    REDO_STACK.push(CURRENT_GRID.clone());
    
    // Restore previous state
    CURRENT_GRID = UNDO_STACK.pop();
    refreshEditorGrid();
    syncFromEditorToData();
    
    updateUndoRedoButtons();
    infoMsg('Undo');
}

function redo() {
    if (REDO_STACK.length === 0) {
        errorMsg('Nothing to redo');
        return;
    }
    
    if (!CURRENT_GRID) return;
    
    // Save current state to undo stack
    UNDO_STACK.push(CURRENT_GRID.clone());
    
    // Restore next state
    CURRENT_GRID = REDO_STACK.pop();
    refreshEditorGrid();
    syncFromEditorToData();
    
    updateUndoRedoButtons();
    infoMsg('Redo');
}

function clearUndoHistory() {
    UNDO_STACK = [];
    REDO_STACK = [];
    updateUndoRedoButtons();
}

function updateUndoRedoButtons() {
    $('#undo_btn').prop('disabled', UNDO_STACK.length === 0);
    $('#redo_btn').prop('disabled', REDO_STACK.length === 0);
}

// ===========================================
// Copy Forward (copy current step to next step)
// ===========================================

function copyForward() {
    let pair = getCurrentPair();
    if (!pair) return;
    
    // Can't copy forward from output (it's the last step)
    if (CURRENT_STEP_INDEX === pair.intermediates.length + 1) {
        errorMsg('Cannot copy forward from OUTPUT - it\'s the final step');
        return;
    }
    
    // Can't copy forward if next step is output and it's locked
    let nextIsOutput = CURRENT_STEP_INDEX === pair.intermediates.length;
    if (nextIsOutput && IS_LOCKED) {
        errorMsg('Next step is OUTPUT which is locked. Unlock to copy forward.');
        return;
    }
    
    syncFromEditorToData();
    
    // Get next step index
    let nextStepIndex = CURRENT_STEP_INDEX + 1;
    
    // Copy current grid to next step
    if (nextStepIndex === pair.intermediates.length + 1) {
        // Copying to output
        pair.output = CURRENT_GRID.clone();
    } else {
        // Copying to intermediate
        pair.intermediates[nextStepIndex - 1] = CURRENT_GRID.clone();
    }
    
    renderTimeline();
    
    // Stay on current step (symmetry with Copy Previous)
    infoMsg('Copied to next step! Use → to move forward.');
}

// ===========================================
// Line Tool
// ===========================================

function handleLineToolClick(cell) {
    if (isCurrentStepLocked()) {
        errorMsg('This step is locked. Click "Unlock" to edit.');
        return;
    }
    
    let x = parseInt(cell.attr('x'));
    let y = parseInt(cell.attr('y'));
    let symbol = getSelectedSymbol();
    
    if (LINE_START_CELL === null) {
        // First click - set start point
        LINE_START_CELL = { x: x, y: y };
        cell.addClass('line-start-marker');
        infoMsg('Line start set. Click another cell to draw line, or press Escape to cancel.');
    } else {
        // Second click - draw line
        saveUndoState();
        drawLine(LINE_START_CELL.x, LINE_START_CELL.y, x, y, symbol);
        
        // Clear line start
        $('.cell').removeClass('line-start-marker');
        LINE_START_CELL = null;
        
        refreshEditorGrid();
        syncFromEditorToData();
        infoMsg('Line drawn!');
    }
}

function drawLine(x0, y0, x1, y1, symbol) {
    // Bresenham's line algorithm
    let dx = Math.abs(x1 - x0);
    let dy = Math.abs(y1 - y0);
    let sx = (x0 < x1) ? 1 : -1;
    let sy = (y0 < y1) ? 1 : -1;
    let err = dx - dy;
    
    while (true) {
        // Set cell at (x0, y0)
        if (x0 >= 0 && x0 < CURRENT_GRID.height && y0 >= 0 && y0 < CURRENT_GRID.width) {
            CURRENT_GRID.grid[x0][y0] = symbol;
        }
        
        if (x0 === x1 && y0 === y1) break;
        
        let e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
}

// ===========================================
// Replace Color
// ===========================================

function showReplaceColorDialog() {
    if (!CURRENT_GRID) {
        errorMsg('No grid loaded');
        return;
    }
    
    $('#replace_color_modal').show();
}

function hideReplaceColorDialog() {
    $('#replace_color_modal').hide();
}

function executeReplaceColor() {
    let fromColor = parseInt($('#replace_from_color').val());
    let toColor = parseInt($('#replace_to_color').val());
    
    if (isNaN(fromColor) || isNaN(toColor)) {
        errorMsg('Invalid color values');
        return;
    }
    
    if (fromColor === toColor) {
        errorMsg('From and To colors are the same');
        return;
    }
    
    if (isCurrentStepLocked()) {
        errorMsg('This step is locked. Click "Unlock" to edit.');
        hideReplaceColorDialog();
        return;
    }
    
    saveUndoState();
    
    // Check if there's a selection
    let selected = $('#editor_grid').find('.ui-selected');
    let count = 0;
    
    if (selected.length > 0) {
        // Replace only in selected cells
        selected.each(function() {
            let x = parseInt($(this).attr('x'));
            let y = parseInt($(this).attr('y'));
            if (CURRENT_GRID.grid[x][y] === fromColor) {
                CURRENT_GRID.grid[x][y] = toColor;
                count++;
            }
        });
        infoMsg(`Replaced ${count} cells in selection: color ${fromColor} → ${toColor}`);
    } else {
        // Replace in entire grid
        for (let i = 0; i < CURRENT_GRID.height; i++) {
            for (let j = 0; j < CURRENT_GRID.width; j++) {
                if (CURRENT_GRID.grid[i][j] === fromColor) {
                    CURRENT_GRID.grid[i][j] = toColor;
                    count++;
                }
            }
        }
        infoMsg(`Replaced ${count} cells: color ${fromColor} → ${toColor}`);
    }
    
    refreshEditorGrid();
    syncFromEditorToData();
    hideReplaceColorDialog();
}

// ===========================================
// Rotation
// ===========================================

function rotateSelection(degrees) {
    if (!CURRENT_GRID) {
        errorMsg('No grid loaded');
        return;
    }
    
    if (isCurrentStepLocked()) {
        errorMsg('This step is locked. Click "Unlock" to edit.');
        return;
    }
    
    let selected = $('#editor_grid').find('.ui-selected');
    if (selected.length === 0) {
        errorMsg('Select cells first (use Select tool)');
        return;
    }
    
    // Get bounds of selection
    let minRow = Infinity, maxRow = -Infinity;
    let minCol = Infinity, maxCol = -Infinity;
    
    selected.each(function() {
        let row = parseInt($(this).attr('x'));
        let col = parseInt($(this).attr('y'));
        minRow = Math.min(minRow, row);
        maxRow = Math.max(maxRow, row);
        minCol = Math.min(minCol, col);
        maxCol = Math.max(maxCol, col);
    });
    
    let height = maxRow - minRow + 1;
    let width = maxCol - minCol + 1;
    
    // Check if selection is rectangular (all cells in bounds are selected)
    if (selected.length !== height * width) {
        errorMsg('Selection must be rectangular for rotation');
        return;
    }
    
    saveUndoState();
    
    // Extract the selected region
    let region = [];
    for (let i = 0; i < height; i++) {
        region[i] = [];
        for (let j = 0; j < width; j++) {
            region[i][j] = CURRENT_GRID.grid[minRow + i][minCol + j];
        }
    }
    
    // Rotate the region
    let rotated;
    if (degrees === 90) {
        // Clockwise: new[j][height-1-i] = old[i][j]
        rotated = [];
        for (let j = 0; j < width; j++) {
            rotated[j] = [];
            for (let i = height - 1; i >= 0; i--) {
                rotated[j][height - 1 - i] = region[i][j];
            }
        }
    } else {
        // Counter-clockwise (-90): new[width-1-j][i] = old[i][j]
        rotated = [];
        for (let j = width - 1; j >= 0; j--) {
            rotated[width - 1 - j] = [];
            for (let i = 0; i < height; i++) {
                rotated[width - 1 - j][i] = region[i][j];
            }
        }
    }
    
    // Check if rotated region fits
    let newHeight = rotated.length;
    let newWidth = rotated[0].length;
    
    if (minRow + newHeight > CURRENT_GRID.height || minCol + newWidth > CURRENT_GRID.width) {
        errorMsg('Rotated selection would exceed grid bounds');
        undo(); // Revert the saved state
        return;
    }
    
    // Clear the original region (fill with 0)
    for (let i = 0; i < height; i++) {
        for (let j = 0; j < width; j++) {
            CURRENT_GRID.grid[minRow + i][minCol + j] = 0;
        }
    }
    
    // Place the rotated region
    for (let i = 0; i < newHeight; i++) {
        for (let j = 0; j < newWidth; j++) {
            CURRENT_GRID.grid[minRow + i][minCol + j] = rotated[i][j];
        }
    }
    
    refreshEditorGrid();
    syncFromEditorToData();
    
    // Re-select the rotated area
    setTimeout(() => {
        $('.cell').removeClass('ui-selected');
        for (let i = 0; i < newHeight; i++) {
            for (let j = 0; j < newWidth; j++) {
                $(`.cell[x="${minRow + i}"][y="${minCol + j}"]`).addClass('ui-selected');
            }
        }
    }, 50);
    
    infoMsg(`Rotated selection ${degrees > 0 ? '+' : ''}${degrees}°`);
}

function flipSelection(direction) {
    if (!CURRENT_GRID) {
        errorMsg('No grid loaded');
        return;
    }
    
    if (isCurrentStepLocked()) {
        errorMsg('This step is locked. Click "Unlock" to edit.');
        return;
    }
    
    let selected = $('#editor_grid').find('.ui-selected');
    if (selected.length === 0) {
        errorMsg('Select cells first (use Select tool)');
        return;
    }
    
    // Get bounds of selection
    let minRow = Infinity, maxRow = -Infinity;
    let minCol = Infinity, maxCol = -Infinity;
    
    selected.each(function() {
        let row = parseInt($(this).attr('x'));
        let col = parseInt($(this).attr('y'));
        minRow = Math.min(minRow, row);
        maxRow = Math.max(maxRow, row);
        minCol = Math.min(minCol, col);
        maxCol = Math.max(maxCol, col);
    });
    
    let height = maxRow - minRow + 1;
    let width = maxCol - minCol + 1;
    
    // Check if selection is rectangular
    if (selected.length !== height * width) {
        errorMsg('Selection must be rectangular for flip');
        return;
    }
    
    saveUndoState();
    
    // Extract the selected region
    let region = [];
    for (let i = 0; i < height; i++) {
        region[i] = [];
        for (let j = 0; j < width; j++) {
            region[i][j] = CURRENT_GRID.grid[minRow + i][minCol + j];
        }
    }
    
    // Flip the region
    if (direction === 'horizontal') {
        // Flip left-right: reverse each row
        for (let i = 0; i < height; i++) {
            region[i].reverse();
        }
    } else {
        // Flip top-bottom: reverse the rows
        region.reverse();
    }
    
    // Place the flipped region back
    for (let i = 0; i < height; i++) {
        for (let j = 0; j < width; j++) {
            CURRENT_GRID.grid[minRow + i][minCol + j] = region[i][j];
        }
    }
    
    refreshEditorGrid();
    syncFromEditorToData();
    
    // Re-select the area
    setTimeout(() => {
        $('.cell').removeClass('ui-selected');
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                $(`.cell[x="${minRow + i}"][y="${minCol + j}"]`).addClass('ui-selected');
            }
        }
    }, 50);
    
    infoMsg(`Flipped selection ${direction === 'horizontal' ? 'horizontally ↔' : 'vertically ↕'}`);
}

// ===========================================
// Pair Preview Modal (Fullscreen)
// ===========================================

// State for pair preview navigation
let PREVIEW_PAIR_TYPE = null;
let PREVIEW_PAIR_INDEX = 0;

function openPairPreview(type, index) {
    if (!CURRENT_TASK) return;
    
    PREVIEW_PAIR_TYPE = type;
    PREVIEW_PAIR_INDEX = index;
    
    renderPairPreview();
    $('#pair_preview_modal').show();
}

function renderPairPreview() {
    if (!CURRENT_TASK) return;
    
    let pairs = PREVIEW_PAIR_TYPE === 'train' ? CURRENT_TASK.train : CURRENT_TASK.test;
    let pair = pairs[PREVIEW_PAIR_INDEX];
    if (!pair) return;
    
    // Update title
    $('#pair_preview_title').text(`🔍 ${PREVIEW_PAIR_TYPE === 'train' ? 'Train' : 'Test'} Pair ${PREVIEW_PAIR_INDEX + 1}`);
    
    // Update counter
    let totalPairs = CURRENT_TASK.train.length + CURRENT_TASK.test.length;
    let currentNum = PREVIEW_PAIR_TYPE === 'train' 
        ? PREVIEW_PAIR_INDEX + 1 
        : CURRENT_TASK.train.length + PREVIEW_PAIR_INDEX + 1;
    $('#pair_preview_counter').text(`${currentNum} / ${totalPairs}`);
    
    // Update nav button states
    let isFirst = PREVIEW_PAIR_TYPE === 'train' && PREVIEW_PAIR_INDEX === 0;
    let isLast = PREVIEW_PAIR_TYPE === 'test' && PREVIEW_PAIR_INDEX === CURRENT_TASK.test.length - 1;
    if (CURRENT_TASK.test.length === 0) {
        isLast = PREVIEW_PAIR_TYPE === 'train' && PREVIEW_PAIR_INDEX === CURRENT_TASK.train.length - 1;
    }
    $('#prev_pair_btn, #prev_pair_btn2').prop('disabled', isFirst);
    $('#next_pair_btn, #next_pair_btn2').prop('disabled', isLast);
    
    // Clear previous content
    $('#preview_input_grid').empty();
    $('#preview_output_grid').empty();
    
    // Calculate canvas size
    let inputMaxDim = Math.max(pair.input.height, pair.input.width);
    let outputMaxDim = Math.max(pair.output.height, pair.output.width);
    let maxDim = Math.max(inputMaxDim, outputMaxDim);
    
    let maxCanvasSize = Math.min(400, window.innerHeight * 0.5);
    let cellSize = Math.max(10, Math.min(40, Math.floor(maxCanvasSize / maxDim)));
    
    // Render input grid
    let inputCanvas = document.createElement('canvas');
    renderGridToCanvas(pair.input, inputCanvas, cellSize * Math.max(pair.input.height, pair.input.width));
    $('#preview_input_grid').append(inputCanvas);
    $('#preview_input_size').text(`${pair.input.height} × ${pair.input.width}`);
    
    // Render output grid
    let outputCanvas = document.createElement('canvas');
    renderGridToCanvas(pair.output, outputCanvas, cellSize * Math.max(pair.output.height, pair.output.width));
    $('#preview_output_grid').append(outputCanvas);
    $('#preview_output_size').text(`${pair.output.height} × ${pair.output.width}`);
}

function prevPairPreview() {
    if (!CURRENT_TASK) return;
    
    if (PREVIEW_PAIR_TYPE === 'train') {
        if (PREVIEW_PAIR_INDEX > 0) {
            PREVIEW_PAIR_INDEX--;
        }
    } else {
        // In test pairs
        if (PREVIEW_PAIR_INDEX > 0) {
            PREVIEW_PAIR_INDEX--;
        } else if (CURRENT_TASK.train.length > 0) {
            // Go to last train pair
            PREVIEW_PAIR_TYPE = 'train';
            PREVIEW_PAIR_INDEX = CURRENT_TASK.train.length - 1;
        }
    }
    renderPairPreview();
}

function nextPairPreview() {
    if (!CURRENT_TASK) return;
    
    if (PREVIEW_PAIR_TYPE === 'train') {
        if (PREVIEW_PAIR_INDEX < CURRENT_TASK.train.length - 1) {
            PREVIEW_PAIR_INDEX++;
        } else if (CURRENT_TASK.test.length > 0) {
            // Go to first test pair
            PREVIEW_PAIR_TYPE = 'test';
            PREVIEW_PAIR_INDEX = 0;
        }
    } else {
        // In test pairs
        if (PREVIEW_PAIR_INDEX < CURRENT_TASK.test.length - 1) {
            PREVIEW_PAIR_INDEX++;
        }
    }
    renderPairPreview();
}

function closePairPreview() {
    $('#pair_preview_modal').hide();
}

// ===========================================
// Timeline Preview Modal (Full Timeline View)
// ===========================================

// State for timeline preview - tracks which pair/example is being shown
let PREVIEW_TL_TYPE = null;
let PREVIEW_TL_INDEX = 0;

function openTimelinePreview(stepIndex) {
    // Open preview for the currently selected pair
    if (!CURRENT_TASK || !SELECTED_PAIR_TYPE) return;
    
    PREVIEW_TL_TYPE = SELECTED_PAIR_TYPE;
    PREVIEW_TL_INDEX = SELECTED_PAIR_INDEX;
    
    renderTimelinePreview();
    $('#timeline_preview_modal').show();
}

function renderTimelinePreview() {
    if (!CURRENT_TASK) return;
    
    let pairs = PREVIEW_TL_TYPE === 'train' ? CURRENT_TASK.train : CURRENT_TASK.test;
    let pair = pairs[PREVIEW_TL_INDEX];
    if (!pair) return;
    
    // Update title
    $('#timeline_preview_title').text(`🔍 ${PREVIEW_TL_TYPE === 'train' ? 'Train' : 'Test'} ${PREVIEW_TL_INDEX + 1} Timeline`);
    
    // Update counter
    let totalPairs = CURRENT_TASK.train.length + CURRENT_TASK.test.length;
    let currentNum = PREVIEW_TL_TYPE === 'train' 
        ? PREVIEW_TL_INDEX + 1 
        : CURRENT_TASK.train.length + PREVIEW_TL_INDEX + 1;
    $('#timeline_preview_counter').text(`${PREVIEW_TL_TYPE === 'train' ? 'Train' : 'Test'} ${PREVIEW_TL_INDEX + 1} (${currentNum} / ${totalPairs})`);
    
    // Update nav button states
    let isFirst = PREVIEW_TL_TYPE === 'train' && PREVIEW_TL_INDEX === 0;
    let isLast = PREVIEW_TL_TYPE === 'test' && PREVIEW_TL_INDEX === CURRENT_TASK.test.length - 1;
    if (CURRENT_TASK.test.length === 0) {
        isLast = PREVIEW_TL_TYPE === 'train' && PREVIEW_TL_INDEX === CURRENT_TASK.train.length - 1;
    }
    $('#prev_timeline_btn, #prev_timeline_btn2').prop('disabled', isFirst);
    $('#next_timeline_btn, #next_timeline_btn2').prop('disabled', isLast);
    
    // Clear container
    let container = $('#timeline_preview_container');
    container.empty();
    
    // Calculate cell size based on grid dimensions and available space
    let maxDim = Math.max(pair.input.height, pair.input.width, pair.output.height, pair.output.width);
    let totalSteps = pair.intermediates.length + 2;
    let availableHeight = window.innerHeight * 0.5;
    let cellSize = Math.max(8, Math.min(25, Math.floor(availableHeight / maxDim)));
    
    // Render INPUT
    container.append(createTimelinePreviewStep('INPUT', pair.input, cellSize, true));
    container.append('<div class="timeline-preview-arrow">→</div>');
    
    // Render intermediate steps
    pair.intermediates.forEach((grid, i) => {
        container.append(createTimelinePreviewStep(`Step ${i + 1}`, grid, cellSize, false));
        container.append('<div class="timeline-preview-arrow">→</div>');
    });
    
    // Render OUTPUT
    container.append(createTimelinePreviewStep('OUTPUT', pair.output, cellSize, true));
}

function createTimelinePreviewStep(label, grid, cellSize) {
    let isLocked = label === 'INPUT' || label === 'OUTPUT';
    let step = $(`
        <div class="timeline-preview-step ${isLocked ? 'locked' : ''}">
            <span class="step-label">${label}${isLocked ? ' 🔒' : ''}</span>
            <div class="step-canvas-wrapper"></div>
        </div>
    `);
    
    let canvas = document.createElement('canvas');
    let canvasSize = cellSize * Math.max(grid.height, grid.width);
    renderGridToCanvas(grid, canvas, canvasSize);
    step.find('.step-canvas-wrapper').append(canvas);
    
    return step;
}

function prevTimelinePreview() {
    if (!CURRENT_TASK) return;
    
    if (PREVIEW_TL_TYPE === 'train') {
        if (PREVIEW_TL_INDEX > 0) {
            PREVIEW_TL_INDEX--;
        }
    } else {
        // In test pairs
        if (PREVIEW_TL_INDEX > 0) {
            PREVIEW_TL_INDEX--;
        } else if (CURRENT_TASK.train.length > 0) {
            // Go to last train pair
            PREVIEW_TL_TYPE = 'train';
            PREVIEW_TL_INDEX = CURRENT_TASK.train.length - 1;
        }
    }
    renderTimelinePreview();
}

function nextTimelinePreview() {
    if (!CURRENT_TASK) return;
    
    if (PREVIEW_TL_TYPE === 'train') {
        if (PREVIEW_TL_INDEX < CURRENT_TASK.train.length - 1) {
            PREVIEW_TL_INDEX++;
        } else if (CURRENT_TASK.test.length > 0) {
            // Go to first test pair
            PREVIEW_TL_TYPE = 'test';
            PREVIEW_TL_INDEX = 0;
        }
    } else {
        // In test pairs
        if (PREVIEW_TL_INDEX < CURRENT_TASK.test.length - 1) {
            PREVIEW_TL_INDEX++;
        }
    }
    renderTimelinePreview();
}

function closeTimelinePreview() {
    $('#timeline_preview_modal').hide();
}

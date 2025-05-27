// Internal state.
var CURRENT_INPUT_GRID = new Grid(3, 3);
var CURRENT_OUTPUT_GRID = new Grid(3, 3);
var TRAIN_PAIRS = new Array();
var TEST_PAIRS = new Array();
var CURRENT_TEST_PAIR_INDEX = 0;
var COPY_PASTE_DATA = new Array();
var ACTIVE_GRID_REFERENCE = null; // To track which grid is in the editor

// Cosmetic.
var EDITION_GRID_HEIGHT = 500;
var EDITION_GRID_WIDTH = 500;
var MAX_CELL_SIZE = 100;

// New function to render a single test pair in the new preview area
function fillTestPairPreview(pairId, inputGrid, outputGrid) {
    var pairSlotId = 'test_pair_preview_' + pairId;
    var pairSlot = $('#' + pairSlotId);
    if (!pairSlot.length) {
        pairSlot = $('<div id="' + pairSlotId + '" class="pair_preview test_pair_item" index="' + pairId + '"></div>');
        pairSlot.appendTo('#test_pairs_preview_area');
    }
    pairSlot.empty();

    // Remove Pair X button (top right)
    var removeTestX = $('<button class="remove-pair-x" title="Remove pair">×</button>');
    removeTestX.on('click', function(e) {
        e.stopPropagation();
        removeTestPair(pairId);
    });
    pairSlot.append(removeTestX);

    // Duplicate Pair button (top right, next to remove)
    var duplicateTestBtn = $('<button class="duplicate-pair-btn" title="Duplicate pair">⧉</button>');
    duplicateTestBtn.on('click', function(e) {
        e.stopPropagation();
        duplicateTestPair(pairId);
    });
    pairSlot.append(duplicateTestBtn);

    // Input grid (clickable)
    var inputContainer = $('<div class="demo_grid_container"></div>');
    var jqInputGrid = $('<div class="input_preview selectable_grid clickable-grid" data-type="test" data-index="' + pairId + '" data-part="input"></div>');
    jqInputGrid.on('click', function(e) {
        loadGridIntoEditor(inputGrid, 'test', pairId, 'input', e.ctrlKey);
    });
    jqInputGrid.appendTo(inputContainer);
    inputContainer.appendTo(pairSlot);
    fillJqGridWithData(jqInputGrid, inputGrid);
    fitCellsToContainer(jqInputGrid, inputGrid.height, inputGrid.width, 200, 200);

    // Output grid (clickable)
    var outputContainer = $('<div class="demo_grid_container"></div>');
    var jqOutputGrid = $('<div class="output_preview selectable_grid clickable-grid" data-type="test" data-index="' + pairId + '" data-part="output"></div>');
    jqOutputGrid.on('click', function(e) {
        loadGridIntoEditor(outputGrid, 'test', pairId, 'output', e.ctrlKey);
    });
    jqOutputGrid.appendTo(outputContainer);
    outputContainer.appendTo(pairSlot);
    fillJqGridWithData(jqOutputGrid, outputGrid);
    fitCellsToContainer(jqOutputGrid, outputGrid.height, outputGrid.width, 200, 200);
}

function resetTask() {
    CURRENT_INPUT_GRID = new Grid(3, 3);
    TRAIN_PAIRS = new Array();
    TEST_PAIRS = new Array();
    CURRENT_TEST_PAIR_INDEX = 0;
    $('#task_preview').html('');
    $('#test_pairs_preview_area').html('');
    showEditorPlaceholder();
    $('#load_task_file_input')[0].value = "";
    display_task_name("No task loaded", null, null);
    $('#workspace').removeClass('task-loaded');
}

function refreshEditionGrid(jqGrid, dataGrid) {
    jqGrid.empty(); // Clear previous content (grid cells or placeholder from showEditorPlaceholder)

    // Assume dataGrid is always valid and renderable when this function is called.
    // Callers (via syncFromDataGridToEditionGrid) are responsible for ensuring CURRENT_OUTPUT_GRID is valid.
    // The placeholder state is managed by showEditorPlaceholder() directly manipulating the DOM.

    fillJqGridWithData(jqGrid, dataGrid);
    setUpEditionGridListeners(jqGrid);
    fitCellsToContainer(jqGrid, dataGrid.height, dataGrid.width, EDITION_GRID_HEIGHT, EDITION_GRID_HEIGHT);
    initializeSelectable();
}

function syncFromEditionGridToDataGrid() {
    copyJqGridToDataGrid($('#output_grid .edition_grid'), CURRENT_OUTPUT_GRID);
}

function syncFromDataGridToEditionGrid() {
    refreshEditionGrid($('#output_grid .edition_grid'), CURRENT_OUTPUT_GRID);
}

function getSelectedSymbol() {
    selected = $('#symbol_picker .selected-symbol-preview')[0];
    return $(selected).attr('symbol');
}

function setUpEditionGridListeners(jqGrid) {
    jqGrid.find('.cell').click(function(event) {
        cell = $(event.target);
        symbol = getSelectedSymbol();

        mode = $('input[name=tool_switching]:checked').val();
        if (mode == 'floodfill') {
            // If floodfill: fill all connected cells.
            syncFromEditionGridToDataGrid();
            grid = CURRENT_OUTPUT_GRID.grid;
            floodfillFromLocation(grid, cell.attr('x'), cell.attr('y'), symbol);
            syncFromDataGridToEditionGrid();
        }
        else if (mode == 'edit') {
            // Else: fill just this cell.
            setCellSymbol(cell, symbol);
        }
    });
}

function resizeOutputGrid() {
    size_str = $('#output_grid_size').val();
    size_tuple = parseSizeTuple(size_str);
    if (!size_tuple) return;

    let new_height = parseInt(size_tuple[0], 10);
    let new_width = parseInt(size_tuple[1], 10);

    if (isNaN(new_height) || new_height <= 0 || new_height > 30 ||
        isNaN(new_width) || new_width <= 0 || new_width > 30) {
        errorMsg("Invalid grid dimensions (1-30).");
        // Restore input field to current actual grid size if input was bad
        if (CURRENT_OUTPUT_GRID && CURRENT_OUTPUT_GRID.height >= 0 && CURRENT_OUTPUT_GRID.width >= 0) { // Check >=0 to handle 0x0 placeholder state
             $('#output_grid_size').val(CURRENT_OUTPUT_GRID.height + 'x' + CURRENT_OUTPUT_GRID.width);
        }
        return;
    }

    let old_grid_content = undefined;

    syncFromEditionGridToDataGrid();
    old_grid_content = JSON.parse(JSON.stringify(CURRENT_OUTPUT_GRID.grid));


    CURRENT_OUTPUT_GRID = new Grid(new_height, new_width, old_grid_content);
    syncFromDataGridToEditionGrid(); // Calls refreshEditionGrid, which will show the new grid.

    $('#output_grid_size').val(CURRENT_OUTPUT_GRID.height + 'x' + CURRENT_OUTPUT_GRID.width);
    enableEditorControls(); // A grid is active after resize
    infoMsg("Grid resized to " + CURRENT_OUTPUT_GRID.height + "x" + CURRENT_OUTPUT_GRID.width + "." + (ACTIVE_GRID_REFERENCE ? " Linked to task part." : " Not linked to task part."));
}

function resetOutputGrid() {

    let reset_height, reset_width;

    reset_height = CURRENT_OUTPUT_GRID.height;
    reset_width = CURRENT_OUTPUT_GRID.width;
    infoMsg("Editor content reset to black (size " + reset_height + "x" + reset_width + " retained). Link to task part (if any) maintained.");

    CURRENT_OUTPUT_GRID = new Grid(reset_height, reset_width); // New grid, all cells default to 0
    syncFromDataGridToEditionGrid(); // This calls refreshEditionGrid, which will show the black grid.
    $('#output_grid_size').val(reset_height + 'x' + reset_width);
    enableEditorControls(); // A grid (black) is now active
}

function showEditorPlaceholder() {
    var jqGrid = $('#output_grid .edition_grid');
    jqGrid.empty();
    jqGrid.append('<div class="editor_placeholder">Select an input or output from a task pair to begin editing.</div>');

    ACTIVE_GRID_REFERENCE = null;
    // CURRENT_OUTPUT_GRID remains as is (e.g., default 3x3), but it's not visually relevant when placeholder is shown.
    // The size input should reflect the hidden CURRENT_OUTPUT_GRID or a default.
    if (!CURRENT_OUTPUT_GRID || CURRENT_OUTPUT_GRID.height === 0) { // If somehow 0x0, reset to default for consistency
        CURRENT_OUTPUT_GRID = new Grid(3,3);
    }
    $('#output_grid_size').val(CURRENT_OUTPUT_GRID.height + 'x' + CURRENT_OUTPUT_GRID.width);

    // Disable controls that don't make sense with a placeholder
    $('#resize_btn').prop('disabled', true);
    $('#output_grid_size').prop('disabled', true);
    // $('#reset_output_grid_btn') needs an ID to be targeted here, assume it's "reset_grid_btn"
    $('#reset_grid_btn').prop('disabled', true); // Assuming button ID is reset_grid_btn
    // Other controls like symbol picker, tools might also be disabled if desired.
}

function enableEditorControls() {
    // Inverse of disabling in showEditorPlaceholder
    $('#resize_btn').prop('disabled', false);
    $('#output_grid_size').prop('disabled', false);
    $('#reset_grid_btn').prop('disabled', false); // Assuming button ID
}

function copyFromInput() {
    syncFromEditionGridToDataGrid();
    updateActiveGridReferenceInMemory(); // Save before overwriting if linked
    CURRENT_OUTPUT_GRID = convertSerializedGridToGridObject(CURRENT_INPUT_GRID.grid); // CURRENT_INPUT_GRID is the current test input
    syncFromDataGridToEditionGrid();
    $('#output_grid_size').val(CURRENT_OUTPUT_GRID.height + 'x' + CURRENT_OUTPUT_GRID.width);
    ACTIVE_GRID_REFERENCE = null; // Copied from test input, not a saved part of a task pair yet.
    initializeSelectable();
    showEditorPlaceholder(); // Set initial editor state when document is ready
}

function fillPairPreview(pairId, inputGrid, outputGrid) {
    var pairSlot = $('#pair_preview_' + pairId);
    if (!pairSlot.length) {
        pairSlot = $('<div id="pair_preview_' + pairId + '" class="pair_preview" index="' + pairId + '"></div>');
        pairSlot.appendTo('#task_preview');
    }
    pairSlot.empty();

    // Remove Pair X button (top right)
    var removeDemoX = $('<button class="remove-pair-x" title="Remove pair">×</button>');
    removeDemoX.on('click', function(e) {
        e.stopPropagation();
        removeDemoPair(pairId);
    });
    pairSlot.append(removeDemoX);

    // Duplicate Pair button (top right, next to remove)
    var duplicateDemoBtn = $('<button class="duplicate-pair-btn" title="Duplicate pair">⧉</button>');
    duplicateDemoBtn.on('click', function(e) {
        e.stopPropagation();
        duplicateDemoPair(pairId);
    });
    pairSlot.append(duplicateDemoBtn);

    // Input grid (clickable)
    var inputContainer = $('<div class="demo_grid_container"></div>');
    var jqInputGrid = $('<div class="input_preview selectable_grid clickable-grid" data-type="train" data-index="' + pairId + '" data-part="input"></div>');
    jqInputGrid.on('click', function(e) {
        loadGridIntoEditor(inputGrid, 'train', pairId, 'input', e.ctrlKey);
    });
    jqInputGrid.appendTo(inputContainer);
    inputContainer.appendTo(pairSlot);
    fillJqGridWithData(jqInputGrid, inputGrid);
    fitCellsToContainer(jqInputGrid, inputGrid.height, inputGrid.width, 200, 200);

    // Output grid (clickable)
    var outputContainer = $('<div class="demo_grid_container"></div>');
    var jqOutputGrid = $('<div class="output_preview selectable_grid clickable-grid" data-type="train" data-index="' + pairId + '" data-part="output"></div>');
    jqOutputGrid.on('click', function(e) {
        loadGridIntoEditor(outputGrid, 'train', pairId, 'output', e.ctrlKey);
    });
    jqOutputGrid.appendTo(outputContainer);
    outputContainer.appendTo(pairSlot);
    fillJqGridWithData(jqOutputGrid, outputGrid);
    fitCellsToContainer(jqOutputGrid, outputGrid.height, outputGrid.width, 200, 200);
}

function loadJSONTask(train, test) {
    resetTask();
    // $('#modal_bg').hide();
    $('#workspace').addClass('task-loaded');
    $('#error_display').hide();
    $('#info_display').hide();
    ACTIVE_GRID_REFERENCE = null;

    TRAIN_PAIRS = [];
    for (var i = 0; i < train.length; i++) {
        let pair = train[i];
        let input_grid = convertSerializedGridToGridObject(pair['input']);
        let output_grid = convertSerializedGridToGridObject(pair['output']);
        TRAIN_PAIRS.push({input: input_grid, output: output_grid});
    }
    renderAllDemoPairs();

    TEST_PAIRS = [];
    $('#test_pairs_preview_area').empty();
    for (var i = 0; i < test.length; i++) {
        let pair = test[i];
        let input_grid = convertSerializedGridToGridObject(pair['input']);
        let output_grid = convertSerializedGridToGridObject(pair['output']);
        TEST_PAIRS.push({input: input_grid, output: output_grid});
    }
    renderAllTestPairs();
}

function display_task_name(task_name_str, task_index, number_of_tasks) {
    big_space = '&nbsp;'.repeat(4);
    document.getElementById('task_name').innerHTML = (
        'Task name:' + big_space + task_name_str + big_space + (
            task_index===null ? '' :
            ( String(task_index) + ' out of ' + String(number_of_tasks) )
        )
    );
}

function loadTaskFromFile(e) {
    var file = e.target.files[0];
    if (!file) {
        errorMsg('No file selected');
        return;
    }
    var reader = new FileReader();
    reader.onload = function(e) {
        var contents = e.target.result;

        try {
            contents = JSON.parse(contents);
            train = contents['train'];
            test = contents['test'];
        } catch (e) {
            errorMsg('Bad file format');
            return;
        }
        loadJSONTask(train, test);
        display_task_name(file.name, null, null);
        $('#load_task_file_input')[0].value = "";
    };
    reader.readAsText(file);
}

function nextTestInput() {
    if (TEST_PAIRS.length == 0 || CURRENT_TEST_PAIR_INDEX >= TEST_PAIRS.length - 1) {
        errorMsg('No next test input. Pick another file?')
        return
    }
    CURRENT_TEST_PAIR_INDEX += 1;
    values = TEST_PAIRS[CURRENT_TEST_PAIR_INDEX]['input'];
    CURRENT_INPUT_GRID = convertSerializedGridToGridObject(values);
    fillTestInput(CURRENT_INPUT_GRID);

    // Display the corresponding test output reference
    var output_values = TEST_PAIRS[CURRENT_TEST_PAIR_INDEX]['output'];
    var output_grid_ref = convertSerializedGridToGridObject(output_values);
    fillTestOutputRef(output_grid_ref);

    $('#current_test_input_id_display').html(CURRENT_TEST_PAIR_INDEX + 1);
    //$('#total_test_input_count_display').html(test.length); // test is not in scope here
    $('#total_test_input_count_display').html(TEST_PAIRS.length);
}

function submitSolution() {
    syncFromEditionGridToDataGrid();
    reference_output = TEST_PAIRS[CURRENT_TEST_PAIR_INDEX]['output'];
    submitted_output = CURRENT_OUTPUT_GRID.grid;
    if (reference_output.length != submitted_output.length) {
        errorMsg('Wrong solution.');
        return
    }
    for (var i = 0; i < reference_output.length; i++){
        ref_row = reference_output[i];
        for (var j = 0; j < ref_row.length; j++){
            if (ref_row[j] != submitted_output[i][j]) {
                errorMsg('Wrong solution.');
                return
            }
        }

    }
    infoMsg('Correct solution!');
}

function fillTestInput(inputGrid) {
    jqInputGrid = $('#evaluation_input');
    fillJqGridWithData(jqInputGrid, inputGrid);
    fitCellsToContainer(jqInputGrid, inputGrid.height, inputGrid.width, 400, 400);
}

function copyToOutput() {
    syncFromEditionGridToDataGrid();
    CURRENT_OUTPUT_GRID = convertSerializedGridToGridObject(CURRENT_INPUT_GRID.grid);
    syncFromDataGridToEditionGrid();
    $('#output_grid_size').val(CURRENT_OUTPUT_GRID.height + 'x' + CURRENT_OUTPUT_GRID.width);
}

function initializeSelectable() {
    var editorGrid = $('#output_grid .edition_grid');
    try {
        // Destroy only on the main editor grid if it was initialized
        if (editorGrid.hasClass('ui-selectable')) {
            editorGrid.selectable('destroy');
        }
    }
    catch (e) {
        // console.error("Error destroying selectable:", e);
    }
    toolMode = $('input[name=tool_switching]:checked').val();
    if (toolMode == 'select') {
        infoMsg('Select some cells and click on a color to fill in, or press C to copy');
        editorGrid.selectable({
            autoRefresh: false,
            filter: '> .row > .cell',
            start: function(event, ui) {
                // Clear previous selections from other grids if any (though not expected now)
                // $('.ui-selected').removeClass('ui-selected');
                // More targeted: only clear selections within this specific grid before starting a new selection action if needed
                // However, jQuery UI selectable usually handles this by replacing the selection.
                // The original code was to prevent issues if multiple .selectable_grid items existed and were selectable.
                // Now that we only target one, this specific start: function might be less critical
                // but let's keep its core intent for cleaning up selections if needed.
                editorGrid.find('.ui-selected').removeClass('ui-selected');
            }
        });
    } else {
        // If not in select mode, ensure selectable is disabled/destroyed if it somehow persisted
        // The destroy call at the beginning should handle this, but an explicit disable might be an option too.
        // if (editorGrid.hasClass('ui-selectable')) { editorGrid.selectable('disable'); }
    }
}

// Initial event binding.
$(document).ready(function () {
    $('#symbol_picker').find('.symbol_preview').click(function(event) {
        symbol_preview = $(event.target);
        $('#symbol_picker').find('.symbol_preview').each(function(i, preview) {
            $(preview).removeClass('selected-symbol-preview');
        })
        symbol_preview.addClass('selected-symbol-preview');

        toolMode = $('input[name=tool_switching]:checked').val();
        if (toolMode == 'select') {
            $('.edition_grid').find('.ui-selected').each(function(i, cell) {
                symbol = getSelectedSymbol();
                setCellSymbol($(cell), symbol);
            });
        }
    });

    $('#output_grid .edition_grid').each(function(i, jqGrid) {
        setUpEditionGridListeners($(jqGrid));
    });

    $('.load_task').on('change', function(event) {
        loadTaskFromFile(event);
    });

    $('.load_task').on('click', function(event) {
      event.target.value = "";
    });

    $('input[type=radio][name=tool_switching]').change(function() {
        initializeSelectable();
    });

    $('input[type=text][name=size]').on('keydown', function(event) {
        if (event.keyCode == 13) {
            resizeOutputGrid();
        }
    });

    $('body').keydown(function(event) {
        // Copy and paste functionality.
        if (event.which == 67) {
            // Press C

            selected = $('.ui-selected');
            if (selected.length == 0) {
                return;
            }

            COPY_PASTE_DATA = [];
            for (var i = 0; i < selected.length; i ++) {
                x = parseInt($(selected[i]).attr('x'));
                y = parseInt($(selected[i]).attr('y'));
                symbol = parseInt($(selected[i]).attr('symbol'));
                COPY_PASTE_DATA.push([x, y, symbol]);
            }
            infoMsg('Cells copied! Select a target cell and press V to paste at location.');

        }
        if (event.which == 86) {
            // Press P
            if (COPY_PASTE_DATA.length == 0) {
                errorMsg('No data to paste.');
                return;
            }
            selected = $('.edition_grid').find('.ui-selected');
            if (selected.length == 0) {
                errorMsg('Select a target cell on the output grid.');
                return;
            }

            jqGrid = $(selected.parent().parent()[0]);

            if (selected.length == 1) {
                targetx = parseInt(selected.attr('x'));
                targety = parseInt(selected.attr('y'));

                xs = new Array();
                ys = new Array();
                symbols = new Array();

                for (var i = 0; i < COPY_PASTE_DATA.length; i ++) {
                    xs.push(COPY_PASTE_DATA[i][0]);
                    ys.push(COPY_PASTE_DATA[i][1]);
                    symbols.push(COPY_PASTE_DATA[i][2]);
                }

                minx = Math.min(...xs);
                miny = Math.min(...ys);
                for (var i = 0; i < xs.length; i ++) {
                    x = xs[i];
                    y = ys[i];
                    symbol = symbols[i];
                    newx = x - minx + targetx;
                    newy = y - miny + targety;
                    res = jqGrid.find('[x="' + newx + '"][y="' + newy + '"] ');
                    if (res.length == 1) {
                        cell = $(res[0]);
                        setCellSymbol(cell, symbol);
                    }
                }
            } else {
                errorMsg('Can only paste at a specific location; only select *one* cell as paste destination.');
            }
        }
    });

    initializeSelectable();
    showEditorPlaceholder(); // Set initial editor state when document is ready
    // Automatically start with a new empty task
    startEmptyTask();

    $(document).on('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && (e.key === 's' || e.key === 'S')) {
            e.preventDefault();
            saveCurrentEditToMemory();
        }
    });
});

// New function to fill the test output reference grid
function fillTestOutputRef(outputGrid) {
    jqOutputGrid = $('#evaluation_output_ref');
    fillJqGridWithData(jqOutputGrid, outputGrid);
    fitCellsToContainer(jqOutputGrid, outputGrid.height, outputGrid.width, 400, 400);
}

// Function to update the in-memory representation (TRAIN_PAIRS or TEST_PAIRS)
// if CURRENT_OUTPUT_GRID was linked to one of them.
function updateActiveGridReferenceInMemory() {
    if (ACTIVE_GRID_REFERENCE) {
        syncFromEditionGridToDataGrid();
        const { type, index, part } = ACTIVE_GRID_REFERENCE;
        if (type === 'train') {
            if (TRAIN_PAIRS[index]) {
                TRAIN_PAIRS[index][part] = new Grid(CURRENT_OUTPUT_GRID.height, CURRENT_OUTPUT_GRID.width, JSON.parse(JSON.stringify(CURRENT_OUTPUT_GRID.grid)));
                if (part === 'input') {
                    fillPairPreview(index, TRAIN_PAIRS[index][part], TRAIN_PAIRS[index]['output']);
                } else {
                    fillPairPreview(index, TRAIN_PAIRS[index]['input'], TRAIN_PAIRS[index][part]);
                }
            }
        } else if (type === 'test') {
            if (TEST_PAIRS[index]) {
                // TEST_PAIRS now stores Grid objects
                TEST_PAIRS[index][part] = new Grid(CURRENT_OUTPUT_GRID.height, CURRENT_OUTPUT_GRID.width, JSON.parse(JSON.stringify(CURRENT_OUTPUT_GRID.grid)));
                // Refresh the specific test pair preview
                if (part === 'input') {
                    fillTestPairPreview(index, TEST_PAIRS[index][part], TEST_PAIRS[index]['output']);
                } else { // part === 'output'
                    fillTestPairPreview(index, TEST_PAIRS[index]['input'], TEST_PAIRS[index][part]);
                }
            }
        }
    }
}

// --- Functions to load grids into the editor ---

function loadGridIntoEditor(grid, type, index, part, saveCurrent=false) {
    syncFromEditionGridToDataGrid();
    if (saveCurrent) {
        updateActiveGridReferenceInMemory();
    }
    ACTIVE_GRID_REFERENCE = { type, index, part };
    CURRENT_OUTPUT_GRID = new Grid(grid.height, grid.width, JSON.parse(JSON.stringify(grid.grid)));
    syncFromDataGridToEditionGrid();
    $('#output_grid_size').val(CURRENT_OUTPUT_GRID.height + 'x' + CURRENT_OUTPUT_GRID.width);
    enableEditorControls(); // Enable controls as a grid is now active
    infoMsg(`Loaded ${type} pair ${index +1} (${part}) into editor.`);
}

function loadDemoInputToEditor(pairId) {
    if (TRAIN_PAIRS[pairId]) {
        loadGridIntoEditor(TRAIN_PAIRS[pairId].input, 'train', pairId, 'input');
    }
}

function loadDemoOutputToEditor(pairId) {
    if (TRAIN_PAIRS[pairId]) {
        loadGridIntoEditor(TRAIN_PAIRS[pairId].output, 'train', pairId, 'output');
    }
}

function loadTestInputToEditor(pairId) {
    if (TEST_PAIRS[pairId]) {
        // TEST_PAIRS now holds Grid objects, so direct access
        loadGridIntoEditor(TEST_PAIRS[pairId].input, 'test', pairId, 'input');
    }
}

function loadTestOutputToEditor(pairId) {
    if (TEST_PAIRS[pairId]) {
        // TEST_PAIRS now holds Grid objects, so direct access
        loadGridIntoEditor(TEST_PAIRS[pairId].output, 'test', pairId, 'output');
    }
}

function copyFromPair() {
    if (!ACTIVE_GRID_REFERENCE) {
        errorMsg('No active grid to copy from pair.');
        return;
    }
    const { type, index, part } = ACTIVE_GRID_REFERENCE;
    let pairArr = (type === 'train') ? TRAIN_PAIRS : (type === 'test' ? TEST_PAIRS : null);
    if (!pairArr || !pairArr[index]) {
        errorMsg('No valid pair found.');
        return;
    }
    let sourceGrid;
    if (part === 'input') {
        sourceGrid = pairArr[index]['output'];
    } else if (part === 'output') {
        sourceGrid = pairArr[index]['input'];
    } else {
        errorMsg('Unknown grid part.');
        return;
    }
    if (!sourceGrid) {
        errorMsg('No paired grid to copy from.');
        return;
    }
    // If sizes do not match, resize the editor grid to match the source
    if (sourceGrid.height !== CURRENT_OUTPUT_GRID.height || sourceGrid.width !== CURRENT_OUTPUT_GRID.width) {
        CURRENT_OUTPUT_GRID = new Grid(sourceGrid.height, sourceGrid.width);
        $('#output_grid_size').val(sourceGrid.height + 'x' + sourceGrid.width);
    }
    // Copy the grid visually (not in memory)
    CURRENT_OUTPUT_GRID = new Grid(sourceGrid.height, sourceGrid.width, JSON.parse(JSON.stringify(sourceGrid.grid)));
    syncFromDataGridToEditionGrid();
    $('#output_grid_size').val(CURRENT_OUTPUT_GRID.height + 'x' + CURRENT_OUTPUT_GRID.width);
    infoMsg('Copied from paired grid.');
}

// --- End of functions to load grids into the editor ---

function prevTestInput() {
    if (TEST_PAIRS.length == 0 || CURRENT_TEST_PAIR_INDEX <= 0) {
        errorMsg('Already at the first test input.');
        return;
    }
    CURRENT_TEST_PAIR_INDEX -= 1;
    values = TEST_PAIRS[CURRENT_TEST_PAIR_INDEX]['input'];
    CURRENT_INPUT_GRID = convertSerializedGridToGridObject(values);
    fillTestInput(CURRENT_INPUT_GRID);

    var output_values = TEST_PAIRS[CURRENT_TEST_PAIR_INDEX]['output'];
    var output_grid_ref = convertSerializedGridToGridObject(output_values);
    fillTestOutputRef(output_grid_ref);

    $('#current_test_input_id_display').html(CURRENT_TEST_PAIR_INDEX + 1);
    $('#total_test_input_count_display').html(TEST_PAIRS.length);
}

// Repurposed from submitSolution - saves the current editor content to its linked pair
function saveCurrentEditToMemory() {
    updateActiveGridReferenceInMemory();
    infoMsg('Current changes in editor saved to memory.');
}

function saveTask() {
    // 1. Prepare train data for JSON
    const train_data_for_json = TRAIN_PAIRS.map(pair => ({
        input: pair.input.grid,
        output: pair.output.grid
    }));

    // 2. Prepare test data for JSON
    const test_data_for_json = TEST_PAIRS.map(pair => ({
        input: pair.input.grid,
        output: pair.output.grid
    }));

    const task_json = {
        train: train_data_for_json,
        test: test_data_for_json
    };

    // 3. Trigger download
    const filename = ($('#task_name').text().replace('Task name:', '').trim() || 'edited_task.json').replace(/\s*\d+\s+out\s+of\s+\d+/g, '').trim();
    const data_str = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(task_json, null, 2));
    const download_anchor_node = document.createElement('a');
    download_anchor_node.setAttribute("href", data_str);
    download_anchor_node.setAttribute("download", filename);
    document.body.appendChild(download_anchor_node); // required for firefox
    download_anchor_node.click();
    download_anchor_node.remove();
    infoMsg('Task saved as ' + filename);
}

function addDemoPair() {
    const defaultInputGrid = new Grid(3, 3);
    const defaultOutputGrid = new Grid(3, 3);
    TRAIN_PAIRS.push({ input: defaultInputGrid, output: defaultOutputGrid });
    renderAllDemoPairs();
    infoMsg('Added new demonstration pair. Edit it and save the task.');
}

function addTestPair() {
    const defaultInputGrid = new Grid(3, 3);
    const defaultOutputGrid = new Grid(3, 3);
    TEST_PAIRS.push({ input: defaultInputGrid, output: defaultOutputGrid });
    renderAllTestPairs();
    infoMsg('Added new test pair. Edit it and save the task.');
}

function removeDemoPair(pairId) {
    if (pairId >= 0 && pairId < TRAIN_PAIRS.length) {
        // If the removed pair was being edited, reset the editor
        if (ACTIVE_GRID_REFERENCE && ACTIVE_GRID_REFERENCE.type === 'train' && ACTIVE_GRID_REFERENCE.index === pairId) {
            resetOutputGrid(); // This sets ACTIVE_GRID_REFERENCE to null and shows placeholder
        }
        TRAIN_PAIRS.splice(pairId, 1);
        renderAllDemoPairs();
        infoMsg('Demonstration pair removed.');
        // Adjust ACTIVE_GRID_REFERENCE if an item after the removed one was active
        if (ACTIVE_GRID_REFERENCE && ACTIVE_GRID_REFERENCE.type === 'train' && ACTIVE_GRID_REFERENCE.index > pairId) {
            ACTIVE_GRID_REFERENCE.index--;
        }
    }
}

function removeTestPair(pairId) {
    if (pairId >= 0 && pairId < TEST_PAIRS.length) {
        if (ACTIVE_GRID_REFERENCE && ACTIVE_GRID_REFERENCE.type === 'test' && ACTIVE_GRID_REFERENCE.index === pairId) {
            resetOutputGrid();
        }
        TEST_PAIRS.splice(pairId, 1);
        renderAllTestPairs();
        infoMsg('Test pair removed.');
        if (ACTIVE_GRID_REFERENCE && ACTIVE_GRID_REFERENCE.type === 'test' && ACTIVE_GRID_REFERENCE.index > pairId) {
            ACTIVE_GRID_REFERENCE.index--;
        }
    }
}

function renderAllDemoPairs() {
    $('#task_preview').empty();
    TRAIN_PAIRS.forEach((pair, index) => {
        fillPairPreview(index, pair.input, pair.output);
    });
}

function renderAllTestPairs() {
    $('#test_pairs_preview_area').empty();
    TEST_PAIRS.forEach((pair, index) => {
        fillTestPairPreview(index, pair.input, pair.output);
    });
}

function startEmptyTask() {
    TRAIN_PAIRS = [];
    TEST_PAIRS = [];
    CURRENT_INPUT_GRID = new Grid(3, 3);
    CURRENT_OUTPUT_GRID = new Grid(3, 3);
    CURRENT_TEST_PAIR_INDEX = 0;
    ACTIVE_GRID_REFERENCE = null;
    $('#task_preview').empty();
    $('#test_pairs_preview_area').empty();
    $('#workspace').addClass('task-loaded');
    $('#error_display').hide();
    $('#info_display').hide();
    const letters = 'abcdefghijklmnopqrstuvwxyz';
    const alphanum = 'abcdefghijklmnopqrstuvwxyz0123456789';
    let name = letters[Math.floor(Math.random() * letters.length)];
    for (let i = 0; i < 7; i++) {
        name += alphanum[Math.floor(Math.random() * alphanum.length)];
    }
    name += ".json";
    display_task_name(name, null, null);
    syncFromDataGridToEditionGrid();
    enableEditorControls();
    $('#output_grid_size').val('3x3');
}

function duplicateDemoPair(pairId) {
    if (TRAIN_PAIRS[pairId]) {
        const orig = TRAIN_PAIRS[pairId];
        const newInput = new Grid(orig.input.height, orig.input.width, JSON.parse(JSON.stringify(orig.input.grid)));
        const newOutput = new Grid(orig.output.height, orig.output.width, JSON.parse(JSON.stringify(orig.output.grid)));
        TRAIN_PAIRS.splice(pairId + 1, 0, { input: newInput, output: newOutput });
        renderAllDemoPairs();
        infoMsg('Demonstration pair duplicated.');
    }
}

function duplicateTestPair(pairId) {
    if (TEST_PAIRS[pairId]) {
        const orig = TEST_PAIRS[pairId];
        const newInput = new Grid(orig.input.height, orig.input.width, JSON.parse(JSON.stringify(orig.input.grid)));
        const newOutput = new Grid(orig.output.height, orig.output.width, JSON.parse(JSON.stringify(orig.output.grid)));
        TEST_PAIRS.splice(pairId + 1, 0, { input: newInput, output: newOutput });
        renderAllTestPairs();
        infoMsg('Test pair duplicated.');
    }
}

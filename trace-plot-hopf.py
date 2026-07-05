# Reproducible Hopf plotting with the matched high-contrast colour scale.
from pathlib import Path

from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

DATA_ROOT = SCRIPT_DIR

# Reproducible order. No duplicated case.
CASES = ['hdiv-hopf', 'lm-hopf', 'mixed-hopf']
TIMES = [0.0, 10.0, 10000.0]
TIME_LABELS = ['0', '10', '10000']
FIELD = 'MagneticField'
IMAGE_RESOLUTION = [646, 1056]

# Reproduce the stronger contrast left by the original trace-plot-hopf pipeline:
# the StreamTracer/Clip data at t=0 narrowed the active LUT to roughly
# 0.01--0.12.  Keep that *linear* range for all subsequent frames.  Thus hdiv
# at t=10000 remains blue (max |B| ~= 0.0248), while lm/mixed reach green/yellow.
FIXED_RANGE = (1.0e-2, 1.2e-1)
TERMINAL_SPEED = 1.0e-4
COLOUR_PRESET = 'Blue - Green - Orange'
OUTPUT_DIR = SCRIPT_DIR / 'plot-hopf' / 'trace'

CAMERA_POSITION = [57.85209399303723, -35.77584817371464, 20.4463916008555]
CAMERA_VIEW_UP = [-0.24782125366802385, 0.14656823695236998, 0.9576546236232992]
CAMERA_VIEW_ANGLE = 20.233585858585858
CAMERA_PARALLEL_SCALE = 18.78098499920792


def set_camera(render_view):
    render_view.CameraPosition = CAMERA_POSITION
    render_view.CameraViewUp = CAMERA_VIEW_UP
    render_view.CameraViewAngle = CAMERA_VIEW_ANGLE
    render_view.CameraParallelScale = CAMERA_PARALLEL_SCALE


def hide_widget(proxy):
    try:
        HideInteractiveWidgets(proxy=proxy)
    except Exception:
        pass


def force_time(t, *pipeline_objects):
    """Push the same time through the full pipeline.

    This is the main fix for the old script: not only the source, but also the
    downstream filters are updated at the same time.  That removes the accidental
    dependence on previous case order.
    """
    scene = GetAnimationScene()
    scene.AnimationTime = t
    for obj in pipeline_objects:
        UpdatePipeline(t, obj)


def apply_fixed_lut(lut):
    """Use the original trace pipeline's fixed linear colour map."""
    lut.ApplyPreset(COLOUR_PRESET, True)
    lut.RescaleTransferFunction(*FIXED_RANGE)
    lut.UseLogScale = 0


def render_case(case):
    # Hard reset so that view/LUT/time state from previous cases cannot leak in.
    ResetSession()
    paraview.simple._DisableFirstRenderCameraReset()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pvd_path = DATA_ROOT / case / 'parker.pvd'
    if not pvd_path.exists():
        raise FileNotFoundError(pvd_path)

    parkerpvd = OpenDataFile(str(pvd_path))

    # IMPORTANT: set time first, before creating StreamTracer/Clip.
    force_time(TIMES[0], parkerpvd)

    pdi = parkerpvd.GetPointDataInformation()
    if not pdi.GetArray(FIELD):
        arrays = [pdi.GetArray(i).GetName() for i in range(pdi.GetNumberOfArrays())]
        raise RuntimeError(f'{case}: missing {FIELD}; available point arrays: {arrays}')

    renderView1 = CreateView('RenderView')
    SetActiveView(renderView1)
    renderView1.OrientationAxesVisibility = 0

    layout1 = CreateLayout(name=f'{case}-Layout')
    AssignViewToLayout(view=renderView1, layout=layout1)
    layout1.SetSize(*IMAGE_RESOLUTION)

    # Source display (hidden later, but useful to initialise colour machinery)
    parkerpvdDisplay = Show(parkerpvd, renderView1, 'UnstructuredGridRepresentation')
    parkerpvdDisplay.Representation = 'Surface'
    ColorBy(parkerpvdDisplay, ('POINTS', FIELD, 'Magnitude'))
    parkerpvdDisplay.SetScalarBarVisibility(renderView1, False)

    magneticFieldLUT = GetColorTransferFunction(FIELD)
    GetOpacityTransferFunction(FIELD)
    GetTransferFunction2D(FIELD)
    apply_fixed_lut(magneticFieldLUT)

    outline1 = Outline(registrationName=f'{case}-Outline', Input=parkerpvd)
    outline1Display = Show(outline1, renderView1, 'GeometryRepresentation')
    outline1Display.Representation = 'Surface'
    outline1Display.AmbientColor = [0.0, 0.0, 0.0]
    outline1Display.DiffuseColor = [0.0, 0.0, 0.0]
    outline1Display.LineWidth = 2.0

    streamTracer1 = StreamTracer(
        registrationName=f'{case}-StreamTracer',
        Input=parkerpvd,
        SeedType='Point Cloud',
    )
    streamTracer1.Vectors = ['POINTS', FIELD]
    streamTracer1.TerminalSpeed = TERMINAL_SPEED
    streamTracer1Display = Show(streamTracer1, renderView1, 'GeometryRepresentation')
    streamTracer1Display.Representation = 'Surface'
    ColorBy(streamTracer1Display, ('POINTS', FIELD, 'Magnitude'))
    streamTracer1Display.SetScalarBarVisibility(renderView1, False)
    Hide(parkerpvd, renderView1)

    clip1 = Clip(registrationName=f'{case}-Clip', Input=streamTracer1)
    clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')
    clip1Display.Representation = 'Surface'
    ColorBy(clip1Display, ('POINTS', FIELD, 'Magnitude'))
    clip1Display.SetScalarBarVisibility(renderView1, False)
    clip1Display.LineWidth = 3.0
    Hide(streamTracer1, renderView1)

    hide_widget(clip1.ClipType)
    hide_widget(streamTracer1.SeedType)

    # Push t=0 through the complete chain once after all filters are created.
    force_time(TIMES[0], parkerpvd, streamTracer1, clip1)
    apply_fixed_lut(magneticFieldLUT)

    renderView1.ResetCamera(True, 0.9)
    set_camera(renderView1)
    renderView1.Update()

    for t, label in zip(TIMES, TIME_LABELS):
        # Critical for reproducibility.
        force_time(t, parkerpvd, streamTracer1, clip1)

        # Re-apply colour settings before every screenshot so the result does not
        # depend on case order or ParaView's internal active LUT state.
        ColorBy(clip1Display, ('POINTS', FIELD, 'Magnitude'))
        apply_fixed_lut(magneticFieldLUT)

        renderView1.Update()
        layout1.SetSize(*IMAGE_RESOLUTION)
        set_camera(renderView1)

        out = OUTPUT_DIR / f'{case}-t={label}.png'
        SaveScreenshot(
            filename=str(out),
            viewOrLayout=renderView1,
            location=16,
            ImageResolution=IMAGE_RESOLUTION,
        )
        print(f'saved {out}')

    Delete(clip1)
    Delete(streamTracer1)
    Delete(outline1)
    Delete(parkerpvd)
    Delete(renderView1)


for case in CASES:
    render_case(case)

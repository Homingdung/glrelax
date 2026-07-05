import os
from pathlib import Path

from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

DATA_ROOT = SCRIPT_DIR
OUTPUT_SUFFIX = os.environ.get('E3_TRACE_SUFFIX', 'e3-alter')
INPUT_SUFFIX = os.environ.get('E3_TRACE_INPUT_SUFFIX', 'e3')
OUTPUT_DIR = SCRIPT_DIR / f'plot-{OUTPUT_SUFFIX.removesuffix("-alter")}' / 'trace'

# Cases to render and the timesteps to snapshot (same method as trace-plot.py)
METHODS = ['hdiv', 'lm', 'mixed']
TIMES = [0.0, 9.99999999999998, 10000.0]
TIME_LABELS = ['0', '10', '10000']


def render_case(method):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case = f'{method}-{INPUT_SUFFIX}'
    output_case = f'{method}-{OUTPUT_SUFFIX}'

    # The current solvers write the magnetic field into parker.pvd.  The old
    # recover.pvd output is no longer produced consistently across methods.
    recoverpvd = OpenDataFile(str(DATA_ROOT / case / 'parker.pvd'))
    UpdatePipeline(0.0, recoverpvd)

    # parker.pvd uses MagneticField in all three methods.  Keep a fallback for
    # older datasets whose magnetic-field array used a different name.
    pdi = recoverpvd.GetPointDataInformation()
    field = 'MagneticField' if pdi.GetArray('MagneticField') else None
    if field is None:
        for i in range(pdi.GetNumberOfArrays()):
            a = pdi.GetArray(i)
            if a.GetNumberOfComponents() == 3:
                field = a.GetName()
                break
    if field is None:
        field = pdi.GetArray(0).GetName()
    print(f'{case}: using vector field "{field}"')

    renderView1 = GetActiveViewOrCreate('RenderView')
    recoverpvdDisplay = Show(recoverpvd, renderView1, 'UnstructuredGridRepresentation')
    recoverpvdDisplay.Representation = 'Surface'

    renderView1.ResetCamera(False, 0.9)
    renderView1.Update()
    renderView1.ResetCamera(True, 0.9)

    # Hide orientation axes
    renderView1.OrientationAxesVisibility = 0

    # set scalar coloring by magnetic field magnitude
    ColorBy(recoverpvdDisplay, ('POINTS', field, 'Magnitude'))
    recoverpvdDisplay.RescaleTransferFunctionToDataRange(True, False)
    recoverpvdDisplay.SetScalarBarVisibility(renderView1, True)

    recoveredMagneticFieldLUT = GetColorTransferFunction(field)
    GetOpacityTransferFunction(field)
    GetTransferFunction2D(field)

    recoverpvdDisplay.SetScalarBarVisibility(renderView1, False)

    # outline box
    outline1 = Outline(registrationName='Outline1', Input=recoverpvd)
    outline1Display = Show(outline1, renderView1, 'GeometryRepresentation')
    outline1Display.Representation = 'Surface'
    renderView1.Update()
    outline1Display.AmbientColor = [0.0, 0.0, 0.0]
    outline1Display.DiffuseColor = [0.0, 0.0, 0.0]
    outline1Display.LineWidth = 2.0

    # stream tracer rendered as tubes
    SetActiveSource(recoverpvd)
    streamTracer1 = StreamTracer(registrationName='StreamTracer1', Input=recoverpvd,
                                 SeedType='Point Cloud')
    streamTracer1.Vectors = ['POINTS', field]
    # Finer integration so the streamlines are sampled densely and look smooth
    # (the default cell-length step gives polygonal kinks on this coarse mesh).
    streamTracer1.IntegratorType = 'Runge-Kutta 4-5'
    streamTracer1.IntegrationStepUnit = 'Cell Length'
    streamTracer1.InitialStepLength = 0.05
    streamTracer1.MinimumStepLength = 0.01
    streamTracer1.MaximumStepLength = 0.1
    streamTracer1.MaximumSteps = 20000
    # Point Cloud seed: use a slightly denser set of field lines for clearer traces.
    streamTracer1.SeedType.NumberOfPoints = 100
    streamTracer1.SeedType.Radius = 5
    streamTracer1Display = Show(streamTracer1, renderView1, 'GeometryRepresentation')
    streamTracer1Display.Representation = 'Surface'
    Hide(recoverpvd, renderView1)
    renderView1.Update()
    streamTracer1Display.LineWidth = 5.0
    # color the streamlines themselves by field magnitude
    ColorBy(streamTracer1Display, ('POINTS', field, 'Magnitude'))
    streamTracer1Display.SetScalarBarVisibility(renderView1, False)
    streamTracer1Display.RenderLinesAsTubes = 1

    # color preset
    recoveredMagneticFieldLUT.ApplyPreset('Blue - Green - Orange', True)
    renderView1.Update()

    # Lock the color scale to the t=0 data range (|B| ~ 0.97..3.14, identical for
    # all cases) and keep it FIXED for every timestep -- same as the hdiv-e3 plots.
    # As the field relaxes toward |B|~1 the lines drop to the blue/grey low end,
    # instead of being re-spread into mid-range green by a per-frame rescale.
    animationScene1 = GetAnimationScene()
    animationScene1.AnimationTime = 0.0
    renderView1.Update()
    streamTracer1Display.RescaleTransferFunctionToDataRange(False, True)

    animationScene1 = GetAnimationScene()
    GetTimeKeeper()
    HideInteractiveWidgets(proxy=streamTracer1.SeedType)
    renderView1.Update()

    layout1 = GetLayout()
    layout1.SetSize(548, 1222)

    # camera placement (same as trace-plot.py)
    renderView1.CameraPosition = [185.31430486225514, -77.45432383923038, 48.753717081734344]
    renderView1.CameraViewUp = [-0.21675106710828693, 0.09309623495262145, 0.971777786299453]
    renderView1.CameraViewAngle = 14.811783960720133
    renderView1.CameraParallelScale = 54.98477307757729

    for t, label in zip(TIMES, TIME_LABELS):
        animationScene1.AnimationTime = t
        renderView1.Update()
        # NOTE: no per-frame rescale -- the color scale stays fixed at the t=0 range
        layout1.SetSize(548, 1222)
        renderView1.CameraPosition = [185.31430486225514, -77.45432383923038, 48.753717081734344]
        renderView1.CameraViewUp = [-0.21675106710828693, 0.09309623495262145, 0.971777786299453]
        renderView1.CameraViewAngle = 14.811783960720133
        renderView1.CameraParallelScale = 54.98477307757729
        out = OUTPUT_DIR / f'{output_case}-t={label}.png'
        SaveScreenshot(filename=str(out), viewOrLayout=renderView1, location=16,
                       ImageResolution=[548, 1222])
        print(f'saved {out}')

    # clean up pipeline before next case
    Delete(streamTracer1)
    Delete(outline1)
    Delete(recoverpvd)


for method in METHODS:
    render_case(method)

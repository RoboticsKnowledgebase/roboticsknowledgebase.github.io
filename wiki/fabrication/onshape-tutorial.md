This tutorial is for MRSD students who are new to CAD and want a practical, robotics-focused introduction to Onshape. By the end, you will be able to create sketches, build solid models, assemble parts, add basic parametric behavior, and export drawings for 3D printing and manufacturing!

## 1: Onshape overview and key concepts
Onshape is a cloud-based, parametric CAD system. Instead of saving files locally, everything lives in the cloud as “Documents” that contain:

- Part Studios: where you create one or more 3D parts

- Assemblies: where you connect parts and define motion

- Drawings: 2D engineering drawings for manufacturing or reports

- Supporting items: imported STEP files, PDFs, images, etc.

**key terms:**
- Sketch: a 2D drawing (lines, arcs, circles) on a plane or planar face

- Feature: an operation like extrude, revolve, fillet, shell, pattern

- Feature tree: the ordered list of features on the left that defines how your part was built

- Planes: reference surfaces (Top, Front, Right, or custom planes) on which you sketch

- Constraints: rules that define geometry relationships (parallel, tangent, equal, etc.)

- Dimensions: numeric values that define size and position

- Parametric: model behavior is driven by parameters (dimensions, variables) rather than hand-drawn geometry

Onshape is also history-based, so if you change an early sketch or dimension, every feature that depends on it will rebuild.

## Before You Start

1. Make an Onshape account (free education or free plan).
2. Use Chrome or Edge for best behavior.
3. Use a mouse with a scroll wheel if you can. CAD without a mouse = pain.

## Onshape Interface & Create a Document

- Log into Onshape.

- Click the big blue “Create” button at top-left → choose Document.

- Name it something like: Onshape Course - FirstOnshape. Click OK.

You’ll see a default Part Studio with: 

- A big empty 3D space in the middle.

- Three planes: Top, Front, Right.

- A feature list/feature tree on the left.

### **Interface**

- Graphics Area (center): the 3D world where your parts appear.

- Feature List (Feature Tree) (left):

- At the top: Origin, Top / Front / Right planes.

- As you model: features like “Sketch 1”, “Extrude 1”, etc.

**Toolbars (top):** 

When no sketch is open → you see 3D feature tools (Extrude, Revolve, Fillet, etc.).

When a sketch is open → you see 2D sketch tools (Line, Circle, Rectangle, etc.).

**Bottom Tabs:**

Each tab is a different “thing” in the document:
Part Studios; Assemblies; Drawings; etc.

**Mouse & View Controls (Very Important!)**


- Rotate (Orbit): Hold right mouse button + drag.
 
- Pan: Hold middle mouse button (wheel button) + drag.
 
- Zoom: Scroll the mouse wheel up/down.
 
- Fit view: Hit F on the keyboard to zoom to fit all visible geometry.

- View normal to plane: Select a plane or face → press N.


**Planes: Top, Front, Right (You will start almost every new part by sketching on one of these planes) **


- Top Plane: like looking down at a table.

- Front Plane: like looking at something from the front.

- Right Plane: looking from the right side.


## 2: Starting a Sketch

### Starting a Sketch

1. Click Sketch (top toolbar). Onshape will ask you to select a plane:

1. Click Top Plane.The view will rotate to look normal to the plane.
   You’ll see: 
   
   - A rectangle boundary representing the sketch “view”.

   - The sketch toolbar at the top (Line, Rectangle, Circle, etc.)

   - A small red-green axis (X and Y).

**You are now in Sketch mode.**
**To exit sketch mode:Click the green checkmark ✔ in the top-left of the graphics area.**


**While in Sketch mode, you can find these tools:**

- Line (L): click start → click end.
 
- Rectangle: Corner rectangle: click start corner → click opposite corner. etc.

- Circle: Center point circle: click center → drag → click radius.

- Arc:3-point arc or tangent arc (you can explore later).

- Centerpoint rectangle, slots, etc. (later you can explore more).



### Dimensions 

Use the Dimension tool (looks like a small number with arrows, or press D):

- Click on an edge of an rectangle you just drew.

- Click again where you want to place the dimension text.
  
  A box appears: type a number (e.g., 50) and press Enter.

- Do this for both width and height, e.g.:

  - Horizontal side → 80 mm; Vertical side → 50 mm
  
**Now you have a precise rectangle: 80 mm × 50 mm.**


### Constraints: Making Geometry "Behave" or how sketch elements relate:

- Coincident: a point lies on a line or another point.
 
- Horizontal / Vertical: a line is horizontal or vertical.
 
- Parallel / Perpendicular: relationship between two lines.

- Equal: two segments are the same length; two circles have the same radius.
 
- Symmetric: two entities are symmetric about a line.

- Tangent: a line or curve touches another curve at one point.

In Onshape, uou rarely need to add every constraint manually since Onshape infers many when you draw. You need to understand them so you can edit or remove them when something behaves weirdly.

### Fully Defined vs Under-defined Sketches

- Black geometry = fully defined (fixed size & position).

- Blue geometry = under-defined (can still move or stretch).

- Red geometry = over-defined / conflicting constraints.

**Your goal: All key sketches should be fully defined.**

**To fully define: Add necessary dimensions. Add missing constraints. Tie your sketch to the origin or major reference axes.**


## 3: Turning 2D Into 3D: Extrude & Revolve.

- Extrude = pushing or pulling a sketch into 3D.

- Revolve = spinning a 2D profile around an axis to get something rotational (like a bowl, wheel, knob).

**Assume you have a rectangle sketch on Top Plane:**

- Finish the sketch (green ✔).

- Click Extrude in the top toolbar.

- In the graphics area: Click the rectangle region (the face) so it highlights. On the left, you’ll see the Extrude dialog:

    - Operation: New (create a new solid).

    - Blind / Symmetric / etc.

    - Depth: type 20 mm.

    - Click ✔.

**You now have a 3D block (80 × 50 × 20 mm).**

### View control practice:

- Rotate around it.

- Press F to zoom to fit.

- Click on faces; see how they highlight.


**Extrude Remove (Cutting Material)**

We’ll cut a slot from the top:

- Click the top face of the block.

- Click Sketch.

**You’re now sketching on that face.**

- Draw a small rectangle near the center.

- Add dimensions (e.g., 30 mm by 10 mm), fully define it.

Now:


- Click Extrude.

- Select the new rectangular sketch region.

- Set Operation to Remove & Set Depth:

- Choose Through All (so it cuts all the way down). Click ✔.

**Now you have a block with a rectangular slot.**

### Revolve (Making Round Parts)

Revolve is used for objects like: Bowls, Wheels, Knobs, Lathe-like shapes

- Start a new sketch on the Front Plane or Right Plane.
 
- Draw a 2D half-cross-section of the shape (like the outline of half a bowl).

- Draw a vertical line that will be your axis of rotation.
 
- Make sure the profile is a closed region (no gaps).
 

Now:

- Click Revolve.

- Select the region you want to revolve.

- Select the axis (the line you drew).
 
- Set angle = 360°, Operation = New (for a new solid).


**You now have a revolved 3D solid.**


## 4: Fillets, Chamfers, Shell, Draft

**We will refine shapes so they look real and manufacturable.**

### Fillets (Rounded Edges) softens edges:

- Click Fillet in the toolbar.

- In the graphics area, click one or more edges.

- Type radius, e.g., 5 mm.


**Fillet edges that people might touch (comfort, aesthetics), its internal corners can be used to reduce stress concentrations. Too many fillets can make models heavy/hard to edit—add them later in your feature tree (near the bottom)!**

**Chamfer (Beveled Edges) creates straight beveled edges instead of curve:**

- Click Chamfer.

- Click edge(s).

- Choose chamfer type (distance-distance, angle-distance, etc.).

- Enter values.

**Remove sharp edges. Create a lead-in for screws or parts to slide in.**

### Shell (Hollowing Parts) makes your solid into a thin-walled hollow body:

- Click Shell.

- Select faces to remove (e.g., top face of a box).

- Enter wall thickness (e.g., 2 mm).


### Draft (Tapered Walls) important in molded parts (so parts eject from molds).

- Click Draft.

- Select faces to draft.

- Select neutral plane or edges.

- Enter angle (e.g., 2°).


**If you’re not designing for molding, you can ignore Draft at first. But it’s good to know it exists.**


## 5: Patterns & Mirror (Repeating Features)

### Mirroring Sketch Geometry

In a sketch:

- Draw some geometry on one side of a line.

- Draw a construction line (select a line → mark as construction) to serve as mirror axis.

- Select the geometry to mirror.

- Click Mirror (sketch tool).

- Select the mirror line.

**This creates symmetric geometry.**

### Mirroring 3D Features

- Create a feature (e.g., an extruded boss).

- Click Mirror (3D toolbar).

- For Entities to mirror: select feature(s) from graphics area or feature tree.
  
  For Mirror plane: select one of:
     - Top / Front / Right
     - A mid-plane you created
     - A planar face.



**Onshape will create mirrored solids/features.**

### Linear Patterns (Arrays) duplicates a feature along directions.

- Click Linear pattern.

- Choose: Part or feature you want to repeat.

- Set direction 1: 

   - Click an edge or direction (e.g., X direction).

   - Enter spacing (e.g., 20 mm).

   - Enter number of instances (e.g., 4).


**Circular Patterns Used for holes around a circle (e.g., bolt circles).**

- Click Circular pattern.

- Select:

   - Feature(s) or faces (e.g., a hole).

   - axis of rotation: A cylindrical edge; An axis line

- Set:

   - Number of instances

   - Total angle (e.g., 360°).



## 6: Multi-Part Modeling in a Single Part Studio

**Onshape’s Part Studio can contain multiple parts. This is powerful because parts can share geometry and relationships easily.**

### One Part vs Multiple Parts

- One Part: Simple parts; no need for others to reference.

- Multiple Parts in One Studio: When parts must fit together precisely (e.g., a box and its lid). When you want to design them “in context” with each other.

### Creating a Second Part

- Imagine you have an enclosure (the box) and you want to design a lid that fits exactly.

- Start in the same Part Studio as your box.

- Create a new sketch on the top edge or some plane.

- Use existing geometry:

    - Use “Use” (Project) tool to project edges from the box into your sketch.

    - Draw the lid profile based on projected edges.

    - After you finish the sketch, Extrude as New:

      In the Extrude dialog, set Operation = New (not Add).

      This tells Onshape to create a new part instead of merging.

**You’ll see Part 1 and Part 2 appear in the feature tree under Parts.**

### Boolean Operations between Parts

You can do:

- Add: combine parts into one.

- Subtract: cut one part using another.

- Intersect: keep only overlapping volume.


**Add merges solids. New keeps them separate. Remove cuts from existing geometry.**



## 7: Assemblies & Mates (Making Things Move): we take parts and put them together in Assemblies.

### Creating an Assembly

- At the bottom, click the + tab → Create Assembly.

- Name it, e.g., “Phone Stand Assembly”. You see an empty assembly environment.


### Inserting Parts into an Assembly

- Click Insert (usually top-left in assembly).

- In the dialog, select:

    - The document’s Part Studio.

    - Then click each part (Base, Back Support) to insert.

    - Click somewhere in the graphics area to place them.

    - Hit ✔.

**They’re now floating in space with six degrees of freedom each (3 translation, 3 rotation).**

### Fixing One Part: Fix one part as the “ground” reference, then mate everything to it.

- Right-click on the Base in the graphics area or Instance list.

- Choose Fix.

**The base now doesn’t move.**

**Mate Connectors (Onshape’s Connection Points) uses Mate Connectors as the basis for mates:**

Think of them as little coordinate systems located on faces, edges, or points. Mates join two mate connectors with certain constraints. Often you don’t manually create connectors; Onshape generates them as you click faces/edges.


## 8: Drawings (2D Engineering Documentation)

We move from 3D to 2D technical drawings suitable for manufacturing.

### Creating a Drawing

- At the bottom, click + → Create Drawing.

- Choose a template: ISO or ANSI. A3, A4, etc. For beginners, any is fine; A3 or A4 landscape is common.

- Click OK.

**You now see: A 2D sheet with a border and title block.**

### Inserting a Part View: A dialog appears to select a model:

- Choose the Part Studio or a specific part.

- Set the first view: Orientation (Front, Top, Right, Isometric). Scale (e.g., 1:1, 1:2).

- Click in the sheet to place the base view.

### You can then project additional views:###

- Hover near the base view and drag:
    - Up for top view.

    - Right for side view.

    - Place them and confirm.

### Dimensioning

- Use the Dimension tool in drawing mode:

- Click an edge or two points.

- Move the mouse to place the dimension.

- Click to place.

**The dimension text appears.**

- Add: Overall length, width, height. Critical feature dimensions (holes, slots, etc.).
  Hole locations (often dimensioned from two perpendicular reference edges).

### Notes & Annotations

You can add:

- Text notes (e.g., “Material: Aluminum 6061”).

- Center marks on circles.

- Centerlines on symmetric features.

**Use these to describe manufacturing requirements.**

### Title Block

At the bottom-right of the sheet (typical):

- Fill in part name, designer, date, scale, etc. Some fields may auto-fill from document properties.

### Exporting

- Right-click in the drawing tab → Export: PDF (for sharing or printing). DWG/DXF (for some manufacturing workflows).

## Summary & Keyboard Cheatsheet:
In this tutorial we walked through how to:

- Navigate the Onshape interface using both mouse and keyboard shortcuts

- Create fully defined sketches using constraints and dimensions

- Turn sketches into 3D parts with extrude, revolve, shell, fillet, and chamfer

- Use patterns, mirrors, and multi-part Part Studios to manage related geometry

- Build assemblies and add mates to define motion

- Generate 2D drawings suitable for fabrication or documentation

Below is the keyboard cheatsheet for faster CAD building:

**General**

F — Zoom to fit

N — Normal to face/plane

Shift + 7 — Isometric view

**Sketch Mode**

L — Line

D — Dimension

O — Toggle construction geometry

S — Sketch toolbox popup

**Features**

Shift + E — Extrude

Shift + R — Revolve

**Editing**

Ctrl/Cmd + Z — Undo

Ctrl/Cmd + Y — Redo

Right-click — Context menu


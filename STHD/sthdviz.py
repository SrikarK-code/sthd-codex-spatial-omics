import argparse
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, CategoricalColorMapper, HoverTool
from STHD import color_palette, train

# CODEX CHANGE: Completely removed the `rasterize` functions. 
# You cannot rasterize irregularly placed cells into a strict 2D image matrix.

def fast_plot(df, cmap, title="STHD_visualization", save_root_dir=""):
    """
    CODEX CHANGE: Replaced the image array plot with a proper spatial scatter plot 
    using continuous X and Y coordinates. 
    """
    if len(save_root_dir) > 0:
        output_file(filename=f"{save_root_dir}/{title}.html", title=title)
        
    cell_types = list(cmap.keys())
    colors = list(cmap.values())
    mapper = CategoricalColorMapper(factors=cell_types, palette=colors)
    
    source = ColumnDataSource(df)
    
    p = figure(title=title, match_aspect=True, tools="pan,wheel_zoom,box_zoom,reset,save", frame_width=1000, frame_height=1000)
    
    p.scatter(x='x', y='y', source=source, size=4, 
              color={'field': 'STHD_pred_ct', 'transform': mapper}, 
              legend_field='STHD_pred_ct')
              
    p.add_tools(HoverTool(tooltips=[("Class", "@STHD_pred_ct")]))
    p.grid.grid_line_color = None
    p.axis.visible = False
    p.y_range.flipped = True
    
    show(p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_path", type=str, required=True)
    parser.add_argument("--title", default="sthd_pred_cell_type", type=str)
    args = parser.parse_args()

    sthdata = train.load_data_with_pdata(args.patch_path)
    # CODEX CHANGE: Extract continuous `x` and `y` instead of `array_row` and `array_col`.
    df = sthdata.adata.obs[["x", "y", "STHD_pred_ct"]]
    
    cmap = color_palette.get_config_colormap(name="colormap_coloncatlas_98")
    fast_plot(df, cmap=cmap, title=args.title, save_root_dir=args.patch_path)
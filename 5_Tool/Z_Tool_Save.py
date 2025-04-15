def show_maps(dict_data, **kwargs):
    """
    description: 绘图函数，支持绘制多个数据集的地图绘图，支持绘制WRF输出数据和各种二维网格数据；基本上所有二维网格数据，都可以用这个来绘图
    @param {dictionary} 需要绘图的数据集，key为数据集名称，value为数据集的参数字典，参数字典包括以下参数：
        param {str} dataset_name，数据集名称,需要是唯一的，不能重复
        param {str} variable_name, 变量名，如PM2.5
        param {numpy array(2D)} grid_x，经度坐标
        param {numpy array(2D)} grid_y，纬度坐标
        param {numpy array(2D)} grid_concentration，浓度值
        param {str} file_name, 数据集所在文件名,可为空
        param {bool} is_delta, 是否是差值图, default=False；用以区分原始图(如模拟值)和差值图(如模拟值-观测值)

    param {dictionary} kwargs,参数字典，包括以下参数：
        @param {str} unit, 单位；使用示例：unit='ug/m3'
        @param {str} cmap, 颜色映射, default='jet'；使用示例：cmap='jet'
        @param {str} projection, 地图投影, default='', 为空时使用PlateCarree投影；使用示例：projection=ccrs.PlateCarree()
        @param {str} boundary_file, 行政边界文件, default='', 为空时不加载行政边界数据；使用示例：boundary_file='data/china_shp/china.shp'
        @param {str} x_title, x轴标题, default='Longitude'；使用示例：x_title='Longitude'
        @param {str} y_title, y轴标题, default='Latitude'；使用示例：y_title='Latitude'
        @param {bool} show_minmax, 是否显示最小最大值, default=True；使用示例：show_minmax=False
        @param {tuple} value_range,默认最小值和最大值 (default_min_value,default_max_value), default=(None,None);使用示例：value_range=(0,100)
        @param {str}或{tuple} panel_layout, 子图布局, default=None, 为空时自动计算布局；tuple时，示例(col,row);为str时，可从esil.panellayout_helper的PaneLayout类中选择，如PaneLayout.ForceSquare；使用示例：panel_layout=(2,2)
        @param {bool} show_original_grid, 是否显示原始网格, default=False；使用示例：show_original_grid=True
        @param {bool} sharex, 是否共享x轴, default=True；使用示例：sharex=False
        @param {bool} sharey, 是否共享y轴, default=True；使用示例：sharey=False
        @param {bool} show_sup_title, 是否显示共享XY轴标题, default=False；使用示例：show_sup_title=True
        @param {bool} show_lonlat, 是否显示经纬度；default=False；使用示例：show_lonlat=True
        @param {bool} show_plot, 是否显示图形, default=True；使用示例：show_plot=False
        @param {dict} points_data, 如数据不为空（None)，则会叠加点位数据到图上, default=None.该字典包括以下参数：
            {
            'lon': {numpy array(1D)} ,点位的经度坐标
            'lat':{numpy array(1D)} ,点位的纬度坐标
            'value':{numpy array(1D)} ,点位的值,如测点的浓度值
            'cmap':颜色映射, default='jet'
            'edgecolor:点位的边框颜色, default='black'
            'symbol_size':点位的大小, default=50
            }；
            使用示例：points_data={'lon':df_monitor['lon'],'lat':df_monitor['lat'],'value':df_monitor['Monitor_Conc'],'cmap':'jet','edgecolor'='black','symbol_size':50}
        @param {bool} is_wrf_out_data, 是否是wrf输出数据, default=False；使用示例：is_wrf_out_data=True
        @param {dict} delta_map_settings, 差值图的参数设定，default=None,示例：
            {
            'cmap':'coolwarm',差值图的colorbar，默认为coolwarm
            'value_range':(default_min_value,default_max_value),#默认最小值和最大值 (default_min_value,default_max_value), default=(None,None)
            'colorbar_ticks_num':None, 颜色条刻度数量, default=None, 为空时自动计算
            'colorbar_ticks_value_format':颜色条刻度值格式, default=None,示例：'.2f':保留2位小数；'.2e' ：采用科学计数法显示，并保留2位小数
            'value_format':数值格式, default=和外层value_format一致,示例：'.2f':保留2位小数；'.2e' ：采用科学计数法显示，并保留2位小数
            }；
            使用示例：delta_map_settings={'cmap':'coolwarm','default_min_value':-1,'default_max_value':1}
        @param {bool} show_dependenct_colorbar, 是否为每个子图显示独立的颜色条, default=False,即所有子图共享一个图例(colorbar);True时，每个子图显示独立的颜色条(colorbar),这种情况多用于有差值图和原始图的情况；使用示例：show_dependenct_colorbar=True
        @param {str} font_name, 字体名称, default=None；使用示例：font_name='Arial'；前提是已经安装了该字体，否则会报错
        @param {bool} show_grid_line, 是否显示网格线, default=True,仅对wrf数据有效,即is_wrf_out_data=True时有效；使用示例：show_grid_line=False
        @param {str} value_format, 数值格式, default=None,示例：'.2f':保留2位小数；'.2e' ：采用科学计数法显示，并保留2位小数；使用示例：value_format='.2f'
        @param {int} xy_title_fontsize, x,y轴标题字体大小, default=10；使用示例：xy_title_fontsize=10
        @param {int} super_xy_title_fontsize, 共享x,y轴标题字体大小, default=15；使用示例：super_xy_title_fontsize=15
        @param {int} title_fontsize, 标题字体大小, default=14；使用示例：title_fontsize=14
        @param {bool} show_domain_mean, 是否在x轴标题中显示域均值, default=True;为False时显示网格的总和值，仅对show_minmax=True时有效;使用示例：show_domain_mean=False
        @param {bool} show_lonlat_with_char, 是否显示经纬度带字母, default=False;为True时，经度和纬度的度分秒显示为字母；使用示例：show_lonlat_with_char=True
        @param {int} colorbar_ticks_num, 颜色条刻度数量, default=None, 为空时自动计算;使用示例：colorbar_ticks_num=5
        @param {str} colorbar_ticks_value_format, 颜色条刻度值格式, default=None,例：'.2f':保留2位小数；'.2e' ：采用科学计数法显示，并保留2位小数;使用示例：colorbar_ticks_value_format='.2f'
        @param {int} xy_ticks_digits, x,y轴刻度值保留小数位数, default=1;示例：xy_ticks_digits=1
        @param {bool} is_tight_layout, 是否自动调整子图布局，使得子图之间的间距合适，同时确保整个图像适合于保存或显示, default=False,
            通常多个子图使用各自colorbar时,即show_dependenct_colorbar=True时,建议设置为True,否则采用False；这个函数为True时，subplots_hspace和subplots_wspace不起作用;
            使用示例：is_tight_layout=False
        @param {float} subplots_hspace, 各个子图之间的垂直间距，子图之间的垂直间距占整个图像高度的比例。hspace的取值范围通常在0到1之间，其中0表示没有间距，1表示子图高度的整个间距；默认值是0.3；仅对is_tight_layout=False时有效；使用示例：subplots_hspace=0.3
        @param {float} subplots_wspace, 设置子图之间的水平间距占整个图像宽度的比例。wspace的取值范围也通常在0到1之间，其中0表示没有水平间距，1表示子图宽度的整个间距;默认值是0.2;仅对is_tight_layout=False时有效；使用示例：subplots_wspace=0.2
        @param {tuple} subplots_fig_size, 子图的大小, default=None,元组，第一个元素为宽度，第二个元素为高度，为None时，根据数据自动计算。使用示例：subplots_fig_size=(6,6)；如果不设置，则根据数据自动计算：自动计算宽度为6，高度为宽度*数据的高度/数据的宽度；如果设置，则根据设置的大小来绘图。
    @return {None or matplotlib.figure} fig
    调用示例：
    1.不存在差值图的情况：
    fig=show_maps(dict_data,unit='ug/m3',cmap='jet', show_lonlat=True,projection=None, boundary_file='',show_original_grid=False)
    2.存在差值图的情况：
    fig=show_maps(dict_data,unit='ug/m3',cmap='jet', show_lonlat=True,projection=None, boundary_file='',show_original_grid=False,
    delta_map_settings={'cmap':'coolwarm','value_range':(-1,1),'value_format':'.2f'})
    3.显示点位数据：
    fig=show_maps(dict_data,unit='ug/m3',cmap='jet', show_lonlat=True,projection=None, boundary_file='',show_original_grid=False,
    points_data={'lon':[116.36,115.46,114.56,113.66,112.76],'lat':[39.92,39.02,38.12,37.22,36.32],'value':[35,40,45,50,55],'cmap':'jet','edgecolor'='black','symbol_size':50}    )
    4.展示wrf输出数据：
    fig=show_maps(dict_data,unit='ug/m3',cmap='jet', show_lonlat=True,projection=proj, boundary_file='',show_original_grid=False,
    is_wrf_out_data=True,show_grid_line=True,value_format='.2f'    )
    """

    # 获取指定键的值，如果键不存在则返回默认值
    unit = kwargs.get("unit", "")
    cmap = kwargs.get("cmap", "jet")
    show_lonlat = kwargs.get("show_lonlat", False)
    projection = kwargs.get("projection", None)
    boundary_file = kwargs.get("boundary_file", "")
    x_title = kwargs.get("x_title", "Longitude")
    y_title = kwargs.get("y_title", "Latitude")
    show_minmax = kwargs.get("show_minmax", True)
    value_range = kwargs.get("value_range", (None, None))
    panel_layout = kwargs.get("panel_layout", None)
    show_original_grid = kwargs.get("show_original_grid", False)
    sharex = kwargs.get("sharex", True)
    sharey = kwargs.get("sharey", True)
    show_sup_title = kwargs.get("show_sup_title", False)
    is_wrf_out_data = kwargs.get("is_wrf_out_data", False)
    points_data = kwargs.get("points_data", None)
    delta_map_settings = kwargs.get("delta_map_settings", None)
    show_dependenct_colorbar = kwargs.get("show_dependenct_colorbar", False)
    show_plot = kwargs.get("show_plot", True)
    font_name = kwargs.get("font_name", None)
    show_grid_line = kwargs.get(
        "show_grid_line", True
    )  # 是否显示网格线,仅对wrf数据有效
    value_format_conc = kwargs.get("value_format", None)
    xy_title_fontsize = kwargs.get("xy_title_fontsize", 10)
    super_xy_title_fontsize = kwargs.get("super_xy_title_fontsize", 15)
    title_fontsize = kwargs.get("title_fontsize", 14)
    show_domain_mean = kwargs.get("show_domain_mean", True)
    show_lonlat_with_char = kwargs.get("show_lonlat_with_char", False)
    # keywords_for_delta= kwargs.get('keywords_for_delta', 'delta')
    colorbar_ticks_num = kwargs.get("colorbar_ticks_num", None)
    colorbar_ticks_value_format = kwargs.get("colorbar_ticks_value_format", None)
    xy_ticks_digits = kwargs.get("xy_ticks_digits", 1)
    is_tight_layout = kwargs.get(
        "is_tight_layout", False
    )  # 自动调整子图布局，使得子图之间的间距合适，同时确保整个图像适合于保存或显示,默认为False。这个函数为True时，subplots_hspace和subplots_wspace不起作用
    subplots_hspace = kwargs.get(
        "subplots_hspace", 0.3
    )  # 各个子图之间的垂直间距，子图之间的垂直间距占整个图像高度的比例。hspace的取值范围通常在0到1之间，其中0表示没有间距，1表示子图高度的整个间距；默认值是0.3；仅对is_tight_layout=False时有效
    subplots_wspace = kwargs.get(
        "subplots_wspace", 0.2
    )  # 设置子图之间的水平间距占整个图像宽度的比例。wspace的取值范围也通常在0到1之间，其中0表示没有水平间距，1表示子图宽度的整个间距;默认值是0.2;仅对is_tight_layout=False时有效
    subplots_fig_size = kwargs.get(
        "subplots_fig_size", None
    )  # 用来设置各个子图的大小，元组，第一个元素为宽度，第二个元素为高度，为None时，根据数据自动计算
    if font_name != None:
        # 设置中文字体为系统中已安装的字体，如SimSun（宋体）、SimHei（黑体）
        plt.rcParams["font.sans-serif"] = [font_name]  # 设置中文字体为宋体
        plt.rcParams["axes.unicode_minus"] = False  # 用于正常显示负号
    data_types = dict_data.keys()
    case_num = len(data_types)
    plot_columns, plot_rows = get_layout_col_row(case_num, panel_layout=panel_layout)
    if projection == None:
        projection = ccrs.PlateCarree()
    origin_projection = ccrs.PlateCarree() if is_wrf_out_data else projection
    width = 6
    first_key = next(iter(dict_data))
    height = (
        math.ceil(
            width
            * dict_data[first_key]["grid_y"].shape[0]
            / dict_data[first_key]["grid_x"].shape[0]
        )
        if not isinstance(dict_data[first_key]["grid_y"], list)
        else width
    )
    if subplots_fig_size != None:
        width, height = subplots_fig_size
    fig, axs = plt.subplots(
        plot_rows,
        plot_columns,
        figsize=(width * plot_columns, height * plot_rows),
        subplot_kw={"projection": projection},
        sharex=sharex,
        sharey=sharey,
    )
    # if boundary_file:
    #     reader = Reader(boundary_file)
    if isinstance(axs, cartopy.mpl.geoaxes.GeoAxesSubplot):
        axs = np.array([axs])
    if show_sup_title == False and plot_rows > 1:
        y_titles = [y_title] * plot_rows
        for ax, row in zip(axs[:, 0], y_titles):
            ax.set_ylabel(row, rotation=90, size=xy_title_fontsize)

    axs = axs.ravel()
    for ax, data_type in zip(axs, data_types):
        dic_sub_data = dict_data[data_type]
        file_name, variable_name, x, y, grid_concentration, is_delta,custom_map_settings = (
            dic_sub_data["file_name"],
            dic_sub_data["variable_name"],
            dic_sub_data["grid_x"],
            dic_sub_data["grid_y"],
            dic_sub_data["grid_concentration"],
            dic_sub_data["is_delta"],
            dic_sub_data["custom_map_settings"]
        )
        min_value, max_value, mean_value, total_value = (
            np.nanmin(grid_concentration),
            np.nanmax(grid_concentration),
            np.nanmean(grid_concentration),
            np.nansum(grid_concentration),
        )
        ax.text(
            0.5,
            1.07,
            f"{data_type} {variable_name}",
            transform=ax.transAxes,
            fontsize=title_fontsize,
            fontweight="bold",
            ha="center",
        )
        default_min_value, default_max_value = value_range
        vmax_conc = (
            np.nanpercentile(grid_concentration, 99.5)
            if default_max_value == None
            else default_max_value
        )
        vmin_conc = (
            np.nanpercentile(grid_concentration, 0.5)
            if default_min_value == None
            else default_min_value
        )
        if (
            vmax_conc == vmin_conc
        ):  # 避免最大值和最小值相等，设置最大值为网格平均值的1.5倍
            vmax_conc = np.mean(grid_concentration) * 1.5
        if is_delta:
            cmap_delta = delta_map_settings.get("cmap", "coolwarm")
            
            default_delta_vmin, default_delta_vmax = delta_map_settings.get(
                "value_range", (None, None)
            )  # ["default_min_value"],delta_map_settings["default_max_value"]
            unit=custom_map_settings.get("unit",unit) 
            default_delta_vmin, default_delta_vmax=custom_map_settings.get("value_range",(default_delta_vmin, default_delta_vmax))
            show_domain_mean_custom=custom_map_settings.get("show_domain_mean",show_domain_mean)
            vmax_delta = (
                np.nanpercentile(grid_concentration, 99.5)
                if default_delta_vmax == None
                else default_delta_vmax
            )
            vmin_delta = (
                np.nanpercentile(grid_concentration, 0.5)
                if default_delta_vmin == None
                else default_delta_vmin
            )
            max_value_delta = np.max([abs(vmax_delta), abs(vmin_delta)])
            if max_value_delta == 0:  # 避免最大值和最小值相等，设置最大值0.1
                max_value_delta = 0.1
            vmin_delta, vmax_delta = -max_value_delta, max_value_delta
            if show_original_grid:
                contour = ax.pcolormesh(
                    x,
                    y,
                    grid_concentration,
                    cmap=cmap_delta,
                    vmin=vmin_delta,
                    vmax=vmax_delta,
                    transform=origin_projection,
                )
            else:
                contour = ax.contourf(
                    x,
                    y,
                    grid_concentration,
                    cmap=cmap_delta,
                    transform=origin_projection,
                    vmin=vmin_delta,
                    vmax=vmax_delta,
                )
            # 添加colorbar到当前子图
            cbar = plt.colorbar(contour, ax=ax, shrink=0.6)
            colorbar_ticks_num_delta = delta_map_settings.get(
                "colorbar_ticks_num", None
            )
            if colorbar_ticks_num_delta is not None:
                cbar.set_ticks(
                    np.linspace(vmin_delta, vmax_delta, num=colorbar_ticks_num_delta)
                )  # 设置颜色条显示的数值个数
            colorbar_ticks_value_format_delta = delta_map_settings.get(
                "colorbar_ticks_value_format", None
            )
            if colorbar_ticks_value_format_delta is not None:
                cbar.ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(
                        lambda x, _: f"{x:{colorbar_ticks_value_format_delta}}"
                    )
                )  # 设置颜色条数值格式
            if unit != "":
                cbar.set_label(f"({unit})", fontweight="bold")  # 设置标签字体加粗
        else:  # 非差值图
            if show_original_grid:
                contour = ax.pcolormesh(
                    x,
                    y,
                    grid_concentration,
                    cmap=cmap,
                    vmin=vmin_conc,
                    vmax=vmax_conc,
                    transform=origin_projection,
                )
            else:
                contour = ax.contourf(
                    x,
                    y,
                    grid_concentration,
                    cmap=cmap,
                    transform=origin_projection,
                    vmin=vmin_conc,
                    vmax=vmax_conc,
                )
            if show_dependenct_colorbar:
                cbar = plt.colorbar(contour, ax=ax, shrink=0.6)
                if colorbar_ticks_num is not None:
                    cbar.set_ticks(
                        np.linspace(vmin_conc, vmax_conc, num=colorbar_ticks_num)
                    )  # 设置颜色条显示的数值个数
                if colorbar_ticks_value_format is not None:
                    cbar.ax.yaxis.set_major_formatter(
                        plt.FuncFormatter(
                            lambda x, _: f"{x:{colorbar_ticks_value_format}}"
                        )
                    )  # 设置颜色条数值格式
                if unit != "":
                    cbar.set_label(f"({unit})", fontweight="bold")  # 设置标签字体加粗

        if (points_data is not None or dic_sub_data.get("points_data", None) is not None) and not is_delta:
            points_data = (
                dic_sub_data.get("points_data", None)
                if dic_sub_data.get("points_data", None) is not None
                else points_data
            )
            symbol_size = points_data.get("symbol_size", 50)
            edgecolor = points_data.get("edgecolor", "black")
            points_cmap = points_data.get("cmap", cmap)
            ax.scatter(
                x=points_data["lon"],
                y=points_data["lat"],
                s=symbol_size,
                c=points_data["value"],
                cmap=points_cmap,
                transform=origin_projection,
                vmin=vmin_conc,
                vmax=vmax_conc,
                edgecolor=edgecolor,
            )
        # 加载行政边界数据（示例使用shapefile文件）
        if boundary_file:
            geometries = get_boundary_geometries(boundary_file)
            # geometries = reader.geometries()
            if geometries is not None:
                enshicity = cfeature.ShapelyFeature(
                    geometries, origin_projection, edgecolor="k", facecolor="none"
                )  # facecolor='none',设置面和线
                ax.add_feature(enshicity, linewidth=0.3)  # 添加市界细节
        else:  # 没有提供boundary_file时，添加在线地图特征
            # 添加地图特征
            ax.add_feature(
                cfeature.COASTLINE, facecolor="none"
            )  # '-' 或 'solid'：实线# '--' 或 'dashed'：虚线# ':' 或 'dotted'：点线# '-.' 或 'dashdot'：点划线
            ax.add_feature(cfeature.BORDERS, linestyle="solid", facecolor="none")
            ax.add_feature(
                cfeature.LAND, edgecolor="black", facecolor="none"
            )  # facecolor='none'表示边界线围起来区域不填充颜色，只绘制边界线；
            ax.add_feature(cfeature.OCEAN, edgecolor="black", facecolor="none")
        min_longitude, max_longitude, min_latitude, max_latitude = (
            round(x.min(), xy_ticks_digits),
            round(x.max(), xy_ticks_digits),
            round(y.min(), xy_ticks_digits),
            round(y.max(), xy_ticks_digits),
        )
        mean_or_total_label = "Total" if not show_domain_mean else "Mean"
        mean_or_total_value = total_value if not show_domain_mean else mean_value
        value_format = (
            delta_map_settings.get("value_format", value_format_conc)
            if is_delta
            else value_format_conc
        )
        if is_wrf_out_data:
            mesh = ax.gridlines(
                draw_labels=True,
                linestyle="--",
                linewidth=0.6,
                alpha=0.5,
                x_inline=False,
                y_inline=False,
                color="k",
                visible=show_grid_line,
            )
            mesh.top_labels = False
            mesh.right_labels = False
            mesh.xformatter = LONGITUDE_FORMATTER
            mesh.yformatter = LATITUDE_FORMATTER
            interval_x = math.ceil((max_longitude - min_longitude) / 10)
            interval_y = math.ceil((max_latitude - min_latitude) / 10)
            mesh.xlocator = mticker.FixedLocator(
                np.arange(min_longitude, max_longitude, interval_x)
            )
            mesh.ylocator = mticker.FixedLocator(
                np.arange(min_latitude, max_latitude, interval_y)
            )
            mesh.xlabel_style = {"size": xy_title_fontsize}
            mesh.ylabel_style = {"size": xy_title_fontsize}
            if show_minmax:
                if value_format is not None:
                    min_max_info = f"Min= {format(min_value, value_format)}, Max= {format(max_value, value_format)}, {mean_or_total_label}={format(mean_or_total_value, value_format)}"
                else:
                    min_max_info = f"Min= {format_data(min_value)}, Max= {format_data(max_value)}, {mean_or_total_label}={format_data(mean_or_total_value)}"
                ax.text(
                    0.5,
                    1.02,
                    min_max_info,
                    transform=ax.transAxes,
                    fontsize=xy_title_fontsize,
                    ha="center",
                )  # , fontweight='bold'
        else:  # 非wrf数据
            if show_grid_line:
                mesh = ax.gridlines(
                    draw_labels=True,
                    linestyle="--",
                    linewidth=0.6,
                    alpha=0.5,
                    x_inline=False,
                    y_inline=False,
                    color="k",
                    visible=show_grid_line,
                )
                mesh.top_labels = False
                mesh.right_labels = False
                mesh.xformatter = LONGITUDE_FORMATTER
                mesh.yformatter = LATITUDE_FORMATTER
                interval_x = math.ceil((max_longitude - min_longitude) / 10)
                interval_y = math.ceil((max_latitude - min_latitude) / 10)
                mesh.xlocator = mticker.FixedLocator(
                    np.arange(min_longitude, max_longitude, interval_x)
                )
                mesh.ylocator = mticker.FixedLocator(
                    np.arange(min_latitude, max_latitude, interval_y)
                )
                mesh.xlabel_style = {"size": xy_title_fontsize}
                mesh.ylabel_style = {"size": xy_title_fontsize}
                if show_minmax:
                    if value_format is not None:
                        min_max_info = f"Min= {format(min_value, value_format)}, Max= {format(max_value, value_format)}, {mean_or_total_label}={format(mean_or_total_value, value_format)}"
                    else:
                        min_max_info = f"Min= {format_data(min_value)}, Max= {format_data(max_value)}, {mean_or_total_label}={format_data(mean_or_total_value)}"
                    ax.text(
                        0.5,
                        1.02,
                        min_max_info,
                        transform=ax.transAxes,
                        fontsize=xy_title_fontsize,
                        ha="center",
                    )  # , fontweight='bold'
            else:
                # 设置y轴范围
                ax.set_ylim(min_latitude, max_latitude)
                # print(min(y), max(y))
                # 设置 x 和 y 轴的 ticks 范围
                x_ticks, y_ticks = [], []
                interval_digit = 2 if xy_ticks_digits < 1 else xy_ticks_digits + 1
                if max_longitude > 180 and show_lonlat:
                    x_ticks = [-180, -120, -60, 0, 60, 120, 180]
                else:
                    interval_x = round(
                        float((max_longitude - min_longitude) / 7), interval_digit
                    )
                    x_ticks = np.round(
                        np.arange(min_longitude, max_longitude + 0.1, interval_x),
                        xy_ticks_digits,
                    )

                interval_y = round(
                    float((max_latitude - min_latitude) / 7), interval_digit
                )
                y_ticks = np.round(
                    np.arange(min_latitude, max_latitude + 0.1, interval_y),
                    xy_ticks_digits,
                )
                # x_ticks[0],x_ticks[-1]=min_longitude,max_longitude
                # y_ticks[0],y_ticks[-1]=min_latitude,max_latitude
                if show_lonlat_with_char:
                    mesh = ax.gridlines(
                        draw_labels=True,
                        linestyle="--",
                        linewidth=0.6,
                        alpha=0.5,
                        x_inline=False,
                        y_inline=False,
                        color="k",
                        visible=show_grid_line,
                    )
                    mesh.top_labels = False
                    mesh.right_labels = False
                    mesh.xformatter = LONGITUDE_FORMATTER
                    mesh.yformatter = LATITUDE_FORMATTER
                else:
                    ax.set_xticks(x_ticks)  # 设置 x 轴的 ticks 范围
                    ax.set_yticks(y_ticks)  # 设置 y 轴的 ticks 范围
                # min_value, max_value, mean_value, total_value = np.nanmin(grid_concentration), np.nanmax(grid_concentration), np.nanmean(grid_concentration), np.nansum(grid_concentration)
                if show_minmax:
                    if value_format is not None:
                        min_max_info = f"Min= {format(min_value, value_format)}, Max= {format(max_value, value_format)}, {mean_or_total_label}={format(mean_or_total_value, value_format)}"
                    else:
                        min_max_info = f"Min= {format_data(min_value)}, Max= {format_data(max_value)}, {mean_or_total_label}={format_data(mean_or_total_value)}"
                    ax.set_xlabel(
                        f"{ '' if show_sup_title else  x_title}\n{min_max_info}"
                    )
                else:
                    plt.xlabel(x_title)

    if show_sup_title:
        fig.supxlabel(
            f"{x_title}", y=0.08, fontsize=super_xy_title_fontsize, fontweight="normal"
        )  # 标签相对于图形的x位置，范围为0到1，默认为0.01，距离底部的距离为0.01，表示留出一小段空白。1表示距离顶部的距离为0
        fig.supylabel(
            f"{y_title}", x=0.08, fontsize=super_xy_title_fontsize, fontweight="normal"
        )

    if is_tight_layout:
        plt.tight_layout()
    else:  # 有时调节子图之间的间距无效，请检查是不是设置了子图的高度和宽度。高度过长但是宽度过小，导致间距中间很多空白，无法调整
        plt.subplots_adjust(
            hspace=subplots_hspace, wspace=subplots_wspace
        )  # 调整子图之间的垂直间距和水平间接

    if not show_dependenct_colorbar and not is_delta:
        # 添加颜色条
        cbar = plt.colorbar(
            contour,
            fraction=0.02,
            pad=0.04,
            label=f"{unit}",
            ax=axs,
            orientation="vertical",
            shrink=0.7,
        )  # 设置图例高度与图像高度相同, orientation='vertical', shrink=0.7

        if colorbar_ticks_num is not None:
            cbar.set_ticks(
                np.linspace(vmin_conc, vmax_conc, num=colorbar_ticks_num)
            )  # 设置颜色条显示的数值个数
        if colorbar_ticks_value_format is not None:
            cbar.ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:{colorbar_ticks_value_format}}")
            )  # 设置颜色条数值格式
        # cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        if unit != "":
            cbar.set_label(f"({unit})", fontweight="bold")  # 设置标签字体加粗

    if show_plot:  # 显示图形
        plt.show()
    return fig
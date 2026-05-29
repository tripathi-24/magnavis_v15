'''
Created on Jan 6, 2025

@author: Admin
'''

import sys, os
import time
import logging
from datetime import datetime, date

import numpy as np

import vtk
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableView, QLabel, QSpacerItem, QSizePolicy, QSplashScreen, QWidget, QDoubleSpinBox, QHBoxLayout, QTextEdit, QSplitter
from PyQt5 import Qt, uic
from PyQt5.QtCore import pyqtSignal, QAbstractTableModel, QModelIndex, QTimer, QDateTime, QMutex, QObject, QThread
from PyQt5.QtGui import QStandardItemModel, QPixmap, QPalette, QColor, QTextCursor
from PyQt5.QtGui import QDrag, QPixmap, QPainter, QCursor
from PyQt5.QtCore import QMimeData, Qt as CoreQt

import numpy as np
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor, vtkCubeAxesActor

import matplotlib.dates as md
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
# from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
# import matplotlib.pyplot as plt
try:
    import mplcursors
    MPLCURSORS_AVAILABLE = True
except ImportError:
    MPLCURSORS_AVAILABLE = False
    print('mplcursors not available - tooltips will not work. Install with: pip install mplcursors')

import pandas as pd
# import libraries
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pygeomag import GeoMag

import uuid
import subprocess

from pickle import NONE
from PyQt5.uic.Compiler.qtproxies import QtWidgets

from data_convert_now import get_timeseries_magnetic_data
# from predictor_ai import GRUPredictor
from Anomaly_detector import AnomalyDetector

APP_BASE = os.path.dirname(__file__)

# print(APP_BASE)

colors = vtkNamedColors()

def load_vtk_file(path: str):
    # reader = vtk.vtkPolyDataReader()
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(path)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    return reader

def getFilePath(title):
    qfd = QFileDialog()
    # Use current working directory or home directory as default (cross-platform)
    path = os.path.expanduser("~")  # Default to user's home directory
    f_type = "csv(*.csv)"
    f = QFileDialog.getOpenFileName(qfd, title, path, f_type)
    return f

api_df = pd.DataFrame() #columns=['time_H', 'mag_H_nT'])
api_df_new = pd.DataFrame() #columns=['time_H', 'mag_H_nT'])
mutex = QMutex()


class SessionDataManager(QObject):
    finished = pyqtSignal()
    updatedData = pyqtSignal()

    def update_api_df(self, session_id, hours=None, start_time=None, new=False):
        logging.info(f'session {session_id} want to fetch new data')
        global api_df
        global api_df_new
        mutex.lock()
        try:
            if new:
                # print('new ...., getting data')
                api_df_new = get_timeseries_magnetic_data(session_id, hours=hours, start_time=start_time)
                if api_df_new is None or api_df_new.empty:
                    api_df_new = pd.DataFrame(columns=['time_H', 'mag_H_nT'])
            else:
                api_df = get_timeseries_magnetic_data(session_id, hours=hours, start_time=start_time)
                if api_df is None or api_df.empty:
                    api_df = pd.DataFrame(columns=['time_H', 'mag_H_nT'])
            logging.info('data fetched from api in non blocking mode')
        except Exception as e:
            logging.error(f'issue fetching api data: {e}')
            if new:
                api_df_new = pd.DataFrame(columns=['time_H', 'mag_H_nT'])
            else:
                api_df = pd.DataFrame(columns=['time_H', 'mag_H_nT'])
        self.updatedData.emit()
        mutex.unlock()
        self.finished.emit()

class Color(QWidget):

    def __init__(self, color):
        super().__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)

class DraggableLabel(QLabel):
    def mousePressEvent(self, event):
        if event.button() == CoreQt.LeftButton:
            self.drag_start_position = event.pos()

    def mouseMoveEvent(self, event):
        if not (event.buttons() & CoreQt.LeftButton):
            return
        if (event.pos() - self.drag_start_position).manhattanLength() < QApplication.startDragDistance():
            return
        drag = QDrag(self)
        mimedata = QMimeData()
        mimedata.setText(self.text())
        drag.setMimeData(mimedata)
        pixmap = QPixmap(self.size())
        painter = QPainter(pixmap)
        painter.drawPixmap(self.rect(), self.grab())
        painter.end()
        drag.setPixmap(pixmap)
        drag.setHotSpot(event.pos())
        drag.exec_(CoreQt.CopyAction | CoreQt.MoveAction)

class DropLabel(QLabel):
    def __init__(self, *args, **kwargs):
        QLabel.__init__(self, *args, **kwargs)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        pos = event.pos()
        text = event.mimeData().text()
        self.setText(text)
        event.acceptProposedAction()


class MapWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        label = DropLabel("drop there",self)
        # label.setGeometry(190, 65, 100, 100)
        # layout = QVBoxLayout()
        label_to_drag1 = DraggableLabel("drag this 1",self)
        # label_to_drag1.setGeometry(10, 10, 100,100)
        label_to_drag2 = DraggableLabel("drag this 2",self)
        
        layout1 = Qt.QHBoxLayout()
        layout2 = Qt.QVBoxLayout()
        layout3 = Qt.QVBoxLayout()

        layout2.addWidget(Color('red'))
        layout2.addWidget(Color('yellow'))
        layout2.addWidget(Color('purple'))
        layout2.addWidget(label_to_drag1)
        layout2.addWidget(label_to_drag2)

        layout1.addLayout( layout2 )
        layout1.addLayout( layout3 )

        # layout1.addWidget(Color('green'))

        layout3.addWidget(Color('orange'))
        layout3.addWidget(Color('green'))
        layout3.addWidget(label)
        self.setLayout(layout1)
        self.show()
    
class PandasModel(QAbstractTableModel):
    """A model to interface a Qt view with pandas dataframe """

    def __init__(self, dataframe: pd.DataFrame, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._dataframe = dataframe

    def rowCount(self, parent=QModelIndex()) -> int:
        """ Override method from QAbstractTableModel

        Return row count of the pandas DataFrame
        """
        if parent == QModelIndex():
            return len(self._dataframe)

        return 0

    def columnCount(self, parent=QModelIndex()) -> int:
        """Override method from QAbstractTableModel

        Return column count of the pandas DataFrame
        """
        if parent == QModelIndex():
            return len(self._dataframe.columns)
        return 0

    def data(self, index: QModelIndex, role=QtCore.Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None

        if role == QtCore.Qt.DisplayRole:
            return str(self._dataframe.iloc[index.row(), index.column()])

        return None

    def headerData(
        self, section: int, orientation: QtCore.Qt.Vertical, role: QtCore.Qt.ItemDataRole
    ):
        """Override method from QAbstractTableModel

        Return dataframe index as vertical header data and columns as horizontal header data.
        """
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return str(self._dataframe.columns[section])

            if orientation == QtCore.Qt.Vertical:
                return str(self._dataframe.index[section])

        return None

timeSerBase, timeSerForm = uic.loadUiType(os.path.join(APP_BASE, 'ui_files', 'ui_MagneticTimeSeriesWidget.ui'))
class MagTimeSeriesWidget(timeSerBase, timeSerForm):
    def __init__(self, app, parent=None):
        super(timeSerBase, self).__init__(parent)
        self.app = app
        self.setupUi(self)
        self.initWidget()
    
    def timerRefreshEvent(self):
        mag_t_df = get_timeseries_magnetic_data()
        # print(mag_t_df)
        t = mag_t_df['time_H'].tolist()
        y_t = mag_t_df['mag_H_nT'].tolist()
        xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
        self.app._static_ax.clear()
        print('plot cleared...')
        self.app._static_ax.xaxis.set_major_formatter(xfmt)
        # self.app.self.magetic_time_ser_canvas.figure.subplots_adjust(bottom=0.3)
        # self.app._static_ax.tick_params(rotation=15) # xticks(rotation=25)
        self.app._static_ax.plot(t, y_t, ".")
        print('plot updated...')
        
    def initWidget(self):
        currentTime = QDateTime.currentDateTime()
        self.dateTimeEdit_2.setDateTime(currentTime)
        default_start_time = currentTime.addDays(-1)
        self.dateTimeEdit.setDateTime(default_start_time)
        refresh_rate_option = self.comboBox_3.currentText()
        refresh_rate = None
        if '20 sec' in refresh_rate_option:
            refresh_rate = 10
        elif '1 min' in refresh_rate_option:
            refresh_rate = 60
        elif '5 min' in refresh_rate_option:
            refresh_rate = 300
        
        # Add Anomaly Detection Threshold Control
        anomaly_layout = QHBoxLayout()
        anomaly_label = QLabel("Anomaly Threshold Multiplier:")
        anomaly_layout.addWidget(anomaly_label)
        
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setMinimum(0.1)
        self.threshold_spinbox.setMaximum(10.0)
        self.threshold_spinbox.setSingleStep(0.1)
        self.threshold_spinbox.setDecimals(2)
        self.threshold_spinbox.setValue(2.5)  # Default value
        self.threshold_spinbox.setToolTip("Higher values = fewer anomalies detected (more strict)\nLower values = more anomalies detected (more sensitive)")
        anomaly_layout.addWidget(self.threshold_spinbox)
        
        # Connect threshold change to update anomaly detector
        self.threshold_spinbox.valueChanged.connect(self.on_threshold_changed)
        
        # Add to the grid layout (add after row 3, column 5)
        self.gridLayout.addLayout(anomaly_layout, 4, 0, 1, 5)

        # Add Training Window Control (minutes; -1 = all loaded data, 0 = predict-only first when checkpoint exists)
        train_window_layout = QHBoxLayout()
        train_window_label = QLabel("Model Training Window (minutes, -1=all, 0=no initial fit):")
        train_window_layout.addWidget(train_window_label)

        self.train_window_spinbox = QDoubleSpinBox()
        self.train_window_spinbox.setMinimum(-1)
        self.train_window_spinbox.setMaximum(100000)  # effectively unbounded
        self.train_window_spinbox.setSingleStep(5)
        self.train_window_spinbox.setDecimals(0)
        self.train_window_spinbox.setValue(-1)  # default = train on all loaded rows
        self.train_window_spinbox.setToolTip(
            "-1: use all loaded historic+realtime rows when training.\n"
            "0: skip initial full-history fit — first predictor run is predict-only if a pretrained/runtime model exists.\n"
            "N≥1: train only on the last N minutes before each training run."
        )
        train_window_layout.addWidget(self.train_window_spinbox)

        self.train_window_spinbox.valueChanged.connect(self.on_train_window_changed)

        # Place below anomaly control
        self.gridLayout.addLayout(train_window_layout, 5, 0, 1, 5)

        # Status badge for training window + time-aware flag
        status_layout = QHBoxLayout()
        self.train_window_status_label = QLabel()
        self.train_window_status_label.setStyleSheet("color: #444;")
        status_layout.addWidget(self.train_window_status_label)
        self.gridLayout.addLayout(status_layout, 6, 0, 1, 5)
        self.update_train_window_status()
        
        # if refresh_rate:
        #     self.timerRefresh = QtCore.QTimer()
        #     self.timerRefresh.timeout.connect(self.timerRefreshEvent)
        #     self.timerRefresh.start(1000*refresh_rate)
        #     print('refresh timer on')
    
    def on_threshold_changed(self, value):
        """Called when user changes the threshold multiplier"""
        if hasattr(self.app, 'anomaly_detector'):
            old_threshold = self.app.anomaly_detector.threshold_multiplier
            self.app.anomaly_detector.threshold_multiplier = value
            self.app.log(f'Anomaly Detection: Threshold multiplier changed from {old_threshold:.2f} to {value:.2f}', level='Info')
            
            # Recalculate threshold with new multiplier
            if len(self.app.anomaly_detector.prediction_errors) >= self.app.anomaly_detector.min_samples_for_threshold:
                mean_error = np.mean(self.app.anomaly_detector.prediction_errors)
                std_error = np.std(self.app.anomaly_detector.prediction_errors)
                new_threshold_value = mean_error + value * std_error
                self.app.anomaly_detector.anomaly_threshold = new_threshold_value
                self.app.log(f'Anomaly Detection: New threshold value calculated: {new_threshold_value:.2f} nT (mean: {mean_error:.2f}, std: {std_error:.2f}, samples: {len(self.app.anomaly_detector.prediction_errors)})', level='Info')
            else:
                # Not enough samples yet - use default or show warning
                current_threshold = self.app.anomaly_detector.anomaly_threshold
                threshold_str = f'{current_threshold:.2f}' if current_threshold is not None else '30.0 (default)'
                self.app.log(f'Anomaly Detection: Not enough samples for threshold calculation. Need {self.app.anomaly_detector.min_samples_for_threshold}, have {len(self.app.anomaly_detector.prediction_errors)}. Using threshold: {threshold_str} nT', level='Warning')
            
            # Re-run anomaly detection with new threshold
            # This will recalculate the threshold using the new multiplier and re-check all data
            self.app._detect_anomalies()
            # Force canvas update to show new anomalies (with small delay to ensure detection completes)
            if hasattr(self.app, '_update_canvas'):
                QTimer.singleShot(200, self.app._update_canvas)

    def on_train_window_changed(self, value):
        """Training window: -1 → all loaded data; 0 → predict-only first run when checkpoint exists; N≥1 → last N min."""
        try:
            if value < 0:
                self.app.train_window_minutes = None
                self.app.log("Training window: all loaded data (-1).", level="Info")
            elif value == 0:
                self.app.train_window_minutes = 0
                self.app.log(
                    "Training window: 0 min → first predictor run is predict-only when a checkpoint/pretrained model exists.",
                    level="Info",
                )
            else:
                self.app.train_window_minutes = int(value)
                self.app.log(f"Training window set to last {int(value)} minutes.", level="Info")
            self.update_train_window_status()
        except Exception as e:
            self.app.log(f'Error setting training window: {e}', level='Error')

    def update_train_window_status(self):
        """Update UI badge to reflect training window minutes and time-aware status."""
        minutes = getattr(self.app, 'train_window_minutes', None)
        if minutes is None:
            window_text = "Training window: all loaded data"
        elif minutes == 0:
            window_text = "Training window: 0 min (predict-only first if checkpoint)"
        else:
            window_text = f"Training window: last {int(minutes)} min"
        # Daily sin/cos time-of-day features are always used; yearly is optional elsewhere.
        self.train_window_status_label.setText(f"{window_text} | Time-of-day features: on (daily sin/cos)")


appBase, appForm = uic.loadUiType(os.path.join(APP_BASE, 'ui_files', 'ui_ApplicationWindow.ui'))

class ApplicationWindow(appBase, appForm):
    SOURCE, _ = range(2) #, TYPE, ACTIVE = range(3)
    
    def __init__(self, app, parent = None):
        super(appBase, self).__init__(parent)
        self.setupUi(self)
        self.threads = []
        self.framework_2_loaded = False
        # self.tab_2.setMinimumHeight(550)
        self._app = app 
        self._app._app_window = self 
        self.model = None
        # WorkbenchHelper.application = self._app
        # WorkbenchHelper.window = self
        
        # self._graphicsTab = MayaviQWidget(app)
        # self._graphicsManager = GraphicsManager()
        
        # self.uiObjectEditorTabWidget.clear()
        #
        # self.uiObjectEditorTabWidget.addTab(self._app._dataSourceManager._data_sourceTab, "Data_source")
        #
        # self.uiGraphicsTabWidget.clear()
        # self.uiGraphicsTabWidget.addTab(self._graphicsTab, 'Visualization')
        self.setTreeView()
        self.setWindowTitle(app.productName)
        # self.show()
        self.connectSlots()
    
    def connectSlots(self):
        self.actionUpload.triggered.connect(self.loadFile)
        self.actionAdd_Time_Series.triggered.connect(self.addTimeSeries)
    
    def createThread(self, session_id, hours, start_time, new):
        thread = QThread()
        worker = SessionDataManager()
        worker.moveToThread(thread)
        thread.started.connect(lambda: worker.update_api_df(session_id, hours, start_time, new))
        worker.updatedData.connect(self.updateData)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        return thread
    
    def updateData(self):
        # self.log('data updated')
        if not self.framework_2_loaded and len(api_df)>0:
            self._app.load_plot_framework_2()
            self.framework_2_loaded = True
        else:
            self._app._update_xydata(force=True)
    
    def startThreads(self, hours, start_time, new):
        self.threads.clear()
        self.threads = [self.createThread(self._app.session_id, hours, start_time, new)]
        for thread in self.threads:
            print('------<<<<  started thread <<<')
            thread.start()

    def addTimeSeries(self):
        try:
            self._app._dataSourceManager.createTimeSeriesSource()
            self.updateTreeView()
        except Exception as e:
            self.log(msg='error adding time series.', error=e, level='Error')
            
    def showCsvData(self, df_in):
        assert df_in is not None
        view = QTableView()
        # view.resize(800, 500)
        view.horizontalHeader().setStretchLastSection(True)
        view.setAlternatingRowColors(True)
        view.setSelectionBehavior(QTableView.SelectRows)
        df_head = df_in.head()
        model = PandasModel(df_head)
        view.setModel(model)
        layout = Qt.QVBoxLayout()
        self.tab.setLayout(layout)
        df_count = len(df_in.index)
        head_count = len(df_head.index)
        line_ht = 20
        nTableHeight = min(df_count, head_count) * line_ht + 100
        view.setMinimumHeight(nTableHeight)
        view.setMaximumHeight(nTableHeight)
        layout.addWidget(view)
        verticalSpacer = QSpacerItem(120, 140, QSizePolicy.Minimum, QSizePolicy.Expanding)
        if head_count < df_count: #(as in most of the cases)
            layout.addWidget(QLabel(f"Showing {head_count} out of {df_count} rows"))
        layout.addItem(verticalSpacer)
    
    def setTreeView(self):
        self.dataView = self.treeView
        self.dataView.setRootIsDecorated(False)
        self.dataView.setAlternatingRowColors(True)
        self.model = self.createMailModel(self)
        self.dataView.setModel(self.model)
    
    def updateTreeView(self):
        sources = self._app._dataSourceManager._world_list + self._app._dataSourceManager._timeseries_list
        for source in sources:
            if source not in self._app._dataSourceManager._added_sources:
                self.addSource(self.model, source_name=source.name, source_type=source.type)
                self._app._dataSourceManager._added_sources.append(source)
        
    def createMailModel(self, parent):
        model = QStandardItemModel(0, 1, parent) # 3, parent)
        model.setHeaderData(self.SOURCE, QtCore.Qt.Horizontal, "Sources")
        # model.setHeaderData(self.TYPE, QtCore.Qt.Horizontal, "Type")
        # model.setHeaderData(self.ACTIVE, QtCore.Qt.Horizontal, "Active")
        return model
    
    def addSource(self, model, source_name, source_type):
        model.insertRow(0)
        model.setData(model.index(0, self.SOURCE), source_name)
        # todo - update prop view based on other information
        # model.setData(model.index(0, self.TYPE), source_type)
        # model.setData(model.index(0, self.ACTIVE), source_active)
    
    def update_visualisation(self):
        # points = vtk.vtkPoints()
        df_inp = self._app._dataSourceManager._world_list[-1].dataFrame
        cols = df_inp.columns.tolist()
        print('df cols',cols)
        df_inp['Altitude'] = df_inp['Altitude'].interpolate(method='linear')
        skip_factor = 10

        df_inp = df_inp.iloc[::skip_factor].reset_index()
        print('df count', df_inp.count())

        x, y, z, mag = df_inp['Longitude'].to_numpy(), df_inp['Latitude'].to_numpy(), df_inp['Altitude'].to_numpy(), df_inp['Mag'].to_numpy()
        num_pts = 100
        # indices = np.arange(0, num_pts, dtype=float) + 0.5
        # phi = np.arccos(1 - 2*indices/num_pts)
        # theta = np.pi * (1 + 5**0.5) * indices
        
        # x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);
        minx, miny, minz = min(x), min(y), min(z)
        x = (x-minx)*1000
        y = (y-miny)*1000
        z = z-minz
        points = np.stack((x, y, z, mag), axis=1)
        # just keep the supported points having non NaN x, y, z
        points = [(float(x_), float(y_), float(z_), float(mag_)) for x_, y_, z_, mag_ in points if np.True_ not in [np.isnan(h) for h in [x_, y_, z_]]]
        

        # points = np.random.array(size=(len(x), 3))
        # for i in range(len(x)):
        #     points[i, 0] = x[i]
        # # z = np.zeros_like(x)
        # # print('xyz', x, y, z)
        # # print(df_inp.columns)
        # # print('alt', df_inp['Altitude'])
        # scalars = vtk.vtkFloatArray()
        # scalars.SetName("Mag")
        # for i in range(len(x)):
        #     # array_point = np.array([x[i]-minx, y[i]-miny, z[i]-minz] )
        #     points.InsertNextPoint((x[i]-minx)*10000, (y[i]-miny)*10000, z[i]-minz)
        #     # points.InsertNextPoint(x[i], y[i], z[i])
        #     scalars.InsertNextValue(mag[i])
        #
        # print('points generated', points)
        # poly = vtk.vtkPolyData()
        # poly.SetPoints(points)
        # poly.GetPointData().SetScalars(scalars)
        
        # # To create surface of a sphere we need to use Delaunay triangulation
        # d2D = vtk.vtkDelaunay2D()
        # d2D.SetInputData( poly ) # This generates a 3D mesh
        #
        # # We need to extract the surface from the 3D mesh
        # dss = vtk.vtkDataSetSurfaceFilter()
        # dss.SetInputConnection( d2D.GetOutputPort() )
        # print('input set')
        # dss.Update()
        
        # Now we have our final polydata
        # The running time of Delaunay triangulation in VTK is \(O(n\log n)\).
        # This is because the triangulation is computed for each set and then the sets are merged along the splitting line
        
        # spherePoly = dss.GetOutput()
        # print('output set')
        
        # glyphFilter = vtk.vtkVertexGlyphFilter()
        # glyphFilter.SetInputData(poly)
        # glyphFilter.Update()
        
        # points2 = 10 * np.random.normal(size=(np.random.randint(100), 3), scale=0.2)
        # self.current_points_to_plot = points2
        self.current_points_to_plot = points #[:10]/100

        self.clear_point_actors()

        # Finally render the canvas with current points
        self.set_points(self.current_points_to_plot)
        
        # self.current_point_actors = self.build_scene(poly)
        # self.ren.AddActor(self.current_point_actors)
        self._app.iren.Render()
        self._app.ren.Render()
        self._app.renwin.Render()

    def clear_point_actors(self):
        if not hasattr(self, 'current_point_actors'):
            pass
        else:
            self.current_point_actors.VisibilityOff()
            self.current_point_actors.ReleaseGraphicsResources(self._app.renwin)
            self._app.ren.RemoveActor(self.current_point_actors)
    
    def set_points(self, coords):
        """This function sets the new set of coordinates on the canvas
        """
        n_tgt = len(coords)
        radii, colors, indices = ApplicationWindow.sphere_prop_to_vtkarray(n_tgt, 0.01, 0)
        
        polydata = vtk.vtkPolyData()
        polydata.GetPointData().AddArray(radii)
        polydata.GetPointData().SetActiveScalars(radii.GetName())
        polydata.GetPointData().AddArray(colors)
        polydata.GetPointData().AddArray(indices)

        points = vtk.vtkPoints()
        points.SetNumberOfPoints(n_tgt)
        mag_scalars = vtk.vtkFloatArray()
        mag_scalars.SetName("mag")
        for i, (x, y, z, mag) in enumerate(coords):
            points.SetPoint(i, x, y, z)
            mag_scalars.InsertNextValue(mag)
        # polydata.GetPointData().SetScalars(mag_scalars)
        polydata.SetPoints(points)

        # Finally update the renderer
        self.current_point_actors = self.build_scene(polydata)
        self._app.ren.AddActor(self.current_point_actors)

        self._app.iren.Render()
        
        
        use_delaunay = False # creating a 2d surface using the points - not working quite well for unordered points, as well as not working for large number of points
        if use_delaunay:
            # # To create surface of a sphere we need to use Delaunay triangulation
            d2D = vtk.vtkDelaunay2D()
            d2D.SetInputData( polydata ) # This generates a 3D mesh
            # We need to extract the surface from the 3D mesh
            dss = vtk.vtkDataSetSurfaceFilter()
            dss.SetInputConnection( d2D.GetOutputPort() )
            print('input set')
            dss.Update()

            # Now we have our final polydata
            # The running time of Delaunay triangulation in VTK is \(O(n\log n)\).
            # This is because the triangulation is computed for each set and then the sets are merged along the splitting line

            dsPoly = dss.GetOutput()
            print('output set', dsPoly)
            sphere_mapper = vtkPolyDataMapper()
            sphere_mapper.SetInputData(dsPoly)

            sphere_actor = vtk.vtkActor()
            sphere_actor.SetMapper(sphere_mapper)
            self._app.ren.AddActor(sphere_actor)
            self._app.iren.Render()
        
        
        
        # glyphFilter = vtk.vtkVertexGlyphFilter()
        # glyphFilter.SetInputData(poly)
        # glyphFilter.Update()
    
    @staticmethod
    def sphere_prop_to_vtkarray(n_sphere, radius, color):
        radii = vtk.vtkFloatArray()
        radii.SetName('radius')
        for _ in range(n_sphere):
            radii.InsertNextTuple1(radius)

        colors = vtk.vtkFloatArray()
        colors.SetName('color')
        for _ in range(n_sphere):
            colors.InsertNextTuple1(color)

        indices = vtk.vtkIntArray()
        indices.SetName('index')
        for idx in range(n_sphere):
            indices.InsertNextTuple1(idx)

        return radii, colors, indices
    
    def build_scene(self, polydata):
        """build a vtkPolyData object for a given frame of the trajectory
        """

        # The rest is for building the point-spheres
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(0, 0, 0)
        sphere.SetRadius(0.2)
        sphere.SetPhiResolution(10)
        sphere.SetThetaResolution(10)

        self.glyph = vtk.vtkGlyph3D()
        self.glyph.GeneratePointIdsOn()
        self.glyph.SetInputData(polydata)
        self.glyph.SetScaleModeToScaleByScalar()
        self.glyph.SetSourceConnection(sphere.GetOutputPort())
        self.glyph.Update()

        sphere_mapper = vtkPolyDataMapper()
        # sphere_mapper.SetLookupTable(self.lut_color)
        sphere_mapper.SetInputConnection(self.glyph.GetOutputPort())
        sphere_mapper.SetScalarModeToUsePointFieldData()
        sphere_mapper.SelectColorArray('color')
        
        ball_actor = vtk.vtkLODActor()
        ball_actor.SetMapper(sphere_mapper)
        ball_actor.GetProperty().SetAmbient(0.2)
        ball_actor.GetProperty().SetDiffuse(0.5)
        ball_actor.GetProperty().SetSpecular(0.3)

        self._picking_domain = ball_actor

        assembly = vtk.vtkAssembly()
        assembly.AddPart(ball_actor)

        return assembly

    def setup_point_picker_framework(self):
        # Text actor for printing the coordinates of the clicked point
        self.actor_title = vtk.vtkTextActor()
        self.actor_title.SetInput('')
        self.actor_title.GetTextProperty().SetFontFamilyToArial()
        self.actor_title.GetTextProperty().BoldOn()
        self.actor_title.GetTextProperty().SetFontSize(12)
        self.actor_title.GetTextProperty().SetColor(1, 0.9, 0.8)
        self.actor_title.SetDisplayPosition(10, 10)
        self._app.ren.AddActor(self.actor_title)
        
        self._app.iren.AddObserver('LeftButtonPressEvent', self.on_pick_left)
        self._app.iren.AddObserver('RightButtonPressEvent', self.on_pick_right)

        self.current_points_to_plot = np.empty((0, 3))

        # self.set_camera_position() #not required imo

    def on_pick_left(self, obj, event=None):
        """Event handler when a point is mouse-picked with the left mouse button
        """
        
        if not hasattr(self, '_picking_domain'):
            return

        # Get the picked position and retrieve the index of the target that was picked from it
        pos = obj.GetEventPosition()

        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.05)

        picker.AddPickList(self._picking_domain)
        picker.PickFromListOn()
        picker.Pick(pos[0], pos[1], 0, self._app.ren)
        pid = picker.GetPointId()
        print('pick event pid', pid)
        if pid > 0:
            idx = int(self.glyph.GetOutput().GetPointData().GetArray('index').GetTuple1(pid))
            x, y, z, mag = self.current_points_to_plot[idx]
            text = f'postion: {(x, y, z)}; mag: {mag}' # Index: {idx};
            print(text)
            self.actor_title.SetInput(text)
    
    def on_pick_right(self, obj, event=None):
        """Clears the the text field when right mouse button is clicked
        """
    
        self.actor_title.SetInput(f'')
    
    def loadFile(self):
        try:
            self.log('upload clicked', level='Debug')
            uploaded_f_name = getFilePath('Load CSV Data: Geo-Magnetic data')
            if uploaded_f_name and type(uploaded_f_name)==tuple:
                f_name = uploaded_f_name[0]
                f_ext = uploaded_f_name[1]
                if f_name:
                    try:
                        if f_name.lower().endswith('.csv'):
                            self._app._dataSourceManager.loadCsv(f_name) # can this happen in different thread to prevent ui freeze? probably yes - need to figure out - todo
                            self.log(f'Loaded file "{f_name}"')
                            self.showCsvData(self._app._dataSourceManager._world_list[-1].dataFrame)
                            self.updateTreeView()
                            self.update_visualisation()
                            
                        else:
                            self.log(f'{"f_name"} is of unsupported file format. Expected *.csv')
                    except Exception as e_in:
                        self.log(f'error loading {f_name}', error=e_in, level='Error')
                        print('e_in', e_in)
                else:
                    self.log(f'no file selected')
        except Exception as e:
            print('error while loading:', e)
    
    def log(self, msg, error=None, level='Info'):
        try:
            print(msg, error, level)
            now = str(datetime.now())
            html_in = self.textEditLog.toHtml()
            # print(html_in[:-24], '---')
            if level in ['Info', 'Debug']:
                if 'Loaded' not in html_in and msg=='Application Loaded and Running':
                    self.textEditLog.setHtml('''<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">
                    <html><head><meta name="qrichtext" content="1" /><style type="text/css">
                    p, li { white-space: pre-wrap; }</style></head>'''+ \
                    f'''<body style=" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;"><p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">{now} : {msg}</p></body></html>''')
                    self._append_log_to_file(now, msg, level, error)
                    return
                self.textEditLog.setHtml(html_in + f'{now} : {msg}')
            if level in ['Error'] or error!= None:
                self.textEditLog.setHtml(html_in + f'{now} : {msg}. Detailed error:<br> {str(error)}')
            # Append to session log file for offline analysis
            self._append_log_to_file(now, msg, level, error)
            # print('-->', self.textEditLog.toHtml())
        except Exception as e:
            print('error while logging:', e)

    def _append_log_to_file(self, now, msg, level='Info', error=None):
        """Append this log line to the session app.log file for offline analysis."""
        try:
            if not getattr(self, '_app', None) or not getattr(self._app, 'session_id', None):
                return
            log_dir = os.path.join(APP_BASE, 'sessions', self._app.session_id)
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'app.log')
            line = msg if isinstance(msg, str) else str(msg)
            if error is not None:
                line = line + " | Error: " + str(error)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"{now}\t[{level}]\t{line}\n")
        except Exception:
            pass  # do not break UI logging if file write fails

class Source(object):
    def __init__(self):
        self.name = 'Source' # automatic uniquify - todo
        self.is_active = True
        self.type = 'undefined'  # 'undefined' / 'timeseries' / 'spacial'

class TimeSeriesSource(object):
    def __init__(self):
        super(TimeSeriesSource, self).__init__()
        self.name = 'TimeSeries - 1'  # automatic uniquify - todo
        self.dataFrame = None
        self.type = 'timeseries'
        
class World(Source):
    def __init__(self):
        super(World, self).__init__()
        self.name = 'World - 1'  # automatic uniquify - todo
        self.dataFrame = None
        self.filename = ''
        self.type = 'spacial'
        
    def loadCsv(self, f_name):
        print('loading world with', f_name)
        try:
            df = pd.read_csv(f_name, skipinitialspace=True) # to read corrupt csv with extra commas in rows than in header, use file based csv reader and create df - todo
            # drop any empty columns (having all rows as NaN)
            df = df.dropna(axis=1, how='all')
            df.index = df.index + 1
            self.filename = f_name
            self.dataFrame = df
            self.type = 'CSV'
            self.name = os.path.basename(f_name).split(".")[0]
        except Exception as e:
            print('error reading csv', f_name)
            raise e
    
class DataSourceManager(object):
    def __init__(self, app=None):
        self._app = app
        #self._allwidgets = app.AllWidgets()
        #self.data_sourceRegionsModel = ComboBoxModel(self._data_sourceRegionTypes)
        self._data_sourceModel = None # Data_sourceModel(self)
        self._data_sourceTab = None # Data_sourceTab(self, app)
        self.updateUI()
        # List of mesh boundary surface names
        self._world_list = []
        self._timeseries_list = []
        self._added_sources = []
        
        self._surfacelist = []
        # List of mesh cell zone names
        self._celllist = []
    
    def loadCsv(self, f_name):
        world = World()
        world.loadCsv(f_name)
        self._world_list.append(world)
        
    
    def updateUI(self):
        if self._data_sourceTab:
            self._data_sourceTab._updatePropView()
    
    def createTimeSeriesSource(self):
        timeSeriesSource = TimeSeriesSource()
        self._timeseries_list.append(timeSeriesSource)
        
        
class Application(QApplication):
    def __init__(self, arg):
        super().__init__(arg)
        """ Application Constructor"""
        self.splash = QSplashScreen(QPixmap(os.path.join(APP_BASE, '..', 'static_observatory_@IITK.png')))
        self.splash.show()
        self.splash.showMessage("\n    Loaded:\n    modules")
        self._version = "0.1" 
        self.productName = "Magnavis"
        self._appDir = os.getcwd() ## not really, name is slightly misleading, can we put APP_BASE?, maybe
        ## Set default area for saving default values
        self._defaultRootDir = "." + self.productName

        # This now should be ApplicationName
        self._defaultDirBase = "." + self.productName
        self.projectName = ""
        self._scratchDir = ""
        self.srcDir     = APP_BASE #os.path.abspath(os.path.dirname(sys.argv[0]))
        self.projectDir = "" #self.preferencesData.projectDir #os.getcwd()
        # Batch / regression harness: pin session folder so drivers can copy app.log predictably.
        _sid_override = os.environ.get("MAGNAVIS_SESSION_ID_OVERRIDE", "").strip()
        self.session_id = _sid_override if _sid_override else str(uuid.uuid4())
        self.predictor_input_file = None
        # Optional predictor training window (minutes): None = all loaded rows; 0 = predict-only first when checkpoint exists (see app.py).
        self.train_window_minutes = None
        
        ## Init Managers
        self._dataSourceManager = DataSourceManager(self)
        # self._postEngine = PostEngine()
        self._currentnode=None
        self._cp=None
        self._iso=None
        self._cutline=None
        self._contour=None
        self.surfaceslist=[]
        self.cellslist=[]

        self.x_t = []
        self.y_mag_t = []
        self.new_x_t = []
        self.new_y_mag_t = []
        
        # class SimplePredict
        # def predict
        # def simple_predict(t_inp_series, mag_inp_series):
        #     t_delta = t_inp_series[-1] - t_inp_series[-2]
        #     t_out_series = [t_inp_series[-1]+t_delta]
        #     mag_out_series = [mag_inp_series[-1]]
        #     return t_out_series, mag_out_series

        # self.predictions = {
        #         'simple': simple_predict,
        #         'ai_model_1': GRUPredictor
        #     }

        self.splash.showMessage("\n    Loaded:\n    modules\n    loading predictor..")
        # self.predictor = GRUPredictor(window_size=5, initial_train_points=3400,
        #                       epochs_per_update=5, learning_rate=0.001, update_training=True)

        self.world_extent = None
        self.map_extent = None
        self.needs_update_lims = False
        
        wnd = self.initViews()
        self.predict_app_started = False
        self.predict_x_t = []
        self.predict_y_t = []
        self.predict_last_t_in_plot = None
        self.prediction_process = None
        self._new_predictions_line = None
        
        # Initialize anomaly detector for real-time anomaly detection
        self.anomaly_detector = AnomalyDetector(
            threshold_multiplier=2.5,
            min_samples_for_threshold=20,
            std_relative_floor=0.02,
        )
        self.anomaly_line = None  # Scatter plot for anomalies on dynamic plot
        self.anomaly_line_static = None  # Scatter plot for anomalies on static plot
        self.anomaly_vertical_lines = []  # List of vertical lines for anomalies on dynamic plot
        self.anomaly_vertical_lines_static = []  # List of vertical lines for anomalies on static plot
        self.anomaly_times = []
        self.anomaly_values = []
        
        self.load_visualization_framework()
        self.splash.showMessage("\n    Loaded:\n    modules\n    visualization")
        self.load_plot_framework() # takes noticeable time for real time computation of magnetic field over latlon grid, move this away in non-blocking thread - todo
        self.splash.showMessage("\n    Loaded:\n    modules\n    visualization\n    maps")
        # self.load_plot_framework_2()
        self.appWin.startThreads(hours=1, start_time=None, new=False)
        self.splash.showMessage("\n    Loaded:\n    modules\n    visualization\n    maps\n    plots")
        self.log(f'Application Loaded and Running', level='Info')
        self.log(f'Session id "{self.session_id}"')
        
        QTimer.singleShot(3000, self.showAppMaximized)
    
    def showAppMaximized(self):
        self.appWin.showMaximized()
        # ApplicationTemp may call splash.finish() earlier so startup prompts are visible;
        # avoid touching a splash that is already closed/deleted.
        try:
            if getattr(self, "splash", None) is not None:
                self.splash.close()
        except Exception:
            pass
        
    def log(self, msg, error=None, level='Info'):
        self.appWin.log(msg, error=error, level=level)

    def _save_data(self, x_t, y_t):
        folder = APP_BASE # r'C:\Users\Admin\eclipse-workspace\magnavis\src'
        filename = 'predict_input.csv'
        if self.session_id:
            folder = os.path.join(folder, 'sessions', self.session_id)
            # Ensure the session directory exists
            os.makedirs(folder, exist_ok=True)
            
            # NEW: Filter out anomalous data points from training dataset
            # Convert anomaly times to a set for efficient lookup
            anomaly_times_set = set()
            if self.anomaly_times and len(self.anomaly_times) > 0:
                # Normalize anomaly times to datetime for comparison
                for anomaly_time in self.anomaly_times:
                    if isinstance(anomaly_time, (datetime, pd.Timestamp)):
                        anomaly_times_set.add(anomaly_time)
                    else:
                        anomaly_times_set.add(pd.to_datetime(anomaly_time))
            
            # Filter out data points that match anomaly timestamps
            filtered_x_t = []
            filtered_y_t = []
            excluded_count = 0
            
            for time_val, mag_val in zip(x_t, y_t):
                # Convert time to datetime for comparison
                time_dt = pd.to_datetime(time_val) if not isinstance(time_val, (datetime, pd.Timestamp)) else time_val
                
                # Check if this timestamp is an anomaly
                if time_dt not in anomaly_times_set:
                    filtered_x_t.append(time_val)
                    filtered_y_t.append(mag_val)
                else:
                    excluded_count += 1
            
            # Create DataFrame with filtered data (anomalies excluded)
            df_save_inp = pd.DataFrame({'x': filtered_x_t, 'y': filtered_y_t})
            inp_file = os.path.join(folder, filename)
            df_save_inp.to_csv(inp_file, index=False)
            
            if excluded_count > 0:
                self.log(f'Saved prediction input file: {inp_file} with {len(df_save_inp)} data points ({excluded_count} anomalies excluded from training)', level='Info')
            else:
                self.log(f'Saved prediction input file: {inp_file} with {len(df_save_inp)} data points', level='Info')
            
            if not self.predict_app_started:
                self.start_prediction_process(inp_file)
                self.predict_app_started = True
            else:
                self.log('Prediction process already started, skipping new start', level='Debug')

    def load_plot_framework(self):
        window = self.appWin
        layout = Qt.QVBoxLayout()
        window.tab_5.setLayout(layout)
        # Ensure tab widget has proper size policy
        window.tab_5.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        static_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        # Configure canvas to fit available space without scrolling
        static_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        static_canvas.setMinimumHeight(300)  # Reasonable minimum
        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.
        
        layout.addWidget(NavigationToolbar(static_canvas, window))
        layout.addWidget(static_canvas)
        # verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        # layout.addItem(verticalSpacer)


        self._static_ax = static_canvas.figure.subplots()
        # t = np.linspace(0, 10, 501)
        # self._static_ax.plot(t, np.tan(t), ".")
        # shapefile = r'data\India_shapefile_with_Kashmir\India_shape\india_ds.shp' #does not contain crs data
        
        # # Read Indian Cities population data
        # data_file_cities = os.path.join(APP_BASE, *r'data\simplemaps_worldcities_basicv1.77\worldcities.csv'.split('\\'))
        # df = pd.read_csv(data_file_cities)
        # df = df[(df['country'] == 'India')]
        # print(df.head())
        #
        # # import street map
        # print(india_map.crs)
        # # designate coordinate system
        # crs = 'EPSG:4326'
        #
        # # zip x and y coordinates into single feature
        # geometry = [Point(xy) for xy in zip(df['lng'], df['lat'])]
        # # create GeoPandas dataframe
        # geo_df = gpd.GeoDataFrame(df, crs=crs, geometry = geometry)
        #
        # print(geo_df.head())
        # print(geo_df.columns)
        # print('indiamap',india_map, india_map.__class__, india_map.columns)
        # print('geometry', india_map['geometry'].to_list()[0])
        # create figure and axes, assign to subplot
        # fig, ax = plt.subplots(figsize=(15,15))
        delta = 2.0
        x = np.arange(65, 100.0, delta)
        y = np.arange(5, 40.0, delta)
        # x = np.arange(-3.0, 3.0, delta)
        # y = np.arange(-2.0, 2.0, delta)
        X, Y = np.meshgrid(x, y)
        # Z1 = np.exp(-X**2 - Y**2)
        # Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
        # Z = (Z1 - Z2) * 2
        Z = np.ones_like(X)
        # geo_mag = GeoMag(coefficients_file=r"C:/Users/Admin/eclipse-workspace/magnavis/src/data/wmm/WMM_HighResolution_2025.COF", high_resolution=True) # obvious issue with module not loading absolute files, can be manually corrected though
        geo_mag = GeoMag(coefficients_file=r"wmm/WMMHR_2025.COF", high_resolution=True)
        for idx, x_ in np.ndenumerate(x):
            for idy, y_ in np.ndenumerate(y):
                result = geo_mag.calculate(glat=y_, glon=x_, alt=0, time=2025.25)
                Z[idx, idy] = float(result.f) # f is total intensity of magnetic field
            
        CS = self._static_ax.contour(X, Y, Z)
        self._static_ax.clabel(CS, fontsize=7, inline_spacing=-10)
        self._static_ax.set_title('India')
        
        shapefile = os.path.join(APP_BASE, *r'data\india_India_Country_Boundary_MAPOG\india_India_Country_Boundary.shp'.split('\\')) # does contain crs data, ie EPSG:4326
        india_map = gpd.read_file(shapefile)
        for poly in india_map['geometry'].to_list()[0].geoms:
            # print('poly', poly.__class__)
            x, y = poly.exterior.xy
            self._static_ax.plot(x, y, alpha=0.4,color='grey')
        
        # add .shp mapfile to axes
        # india_map.plot(ax=ax, alpha=0.4,color='grey')
        # plt.show()
        
    def load_plot_framework_2(self):
        try:
            window = self.appWin
            
            # print(window.tab_2, window.tab_2.__class__)
            # Use flexible figure size that adapts to container
            static_canvas = FigureCanvas(Figure(figsize=(6, 3)))
            static_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            # Ideally one would use self.addToolBar here, but it is slightly
            # incompatible between PyQt6 and other bindings, so we just add the
            # toolbar as a plain widget instead.
            
            timeSerWidget = MagTimeSeriesWidget(self, self.appWin)
            layout_outer = Qt.QVBoxLayout()
            window.tab_2.setLayout(layout_outer)
            # Ensure tab widget has proper size policy
            window.tab_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout_outer.addWidget(timeSerWidget)
            
            # Store reference to time series widget for access later
            self.timeSerWidget = timeSerWidget
            
            # Sync threshold spinbox with anomaly detector value after widget is fully initialized
            # Use QTimer to ensure widget is ready
            QTimer.singleShot(100, lambda: self._sync_threshold_spinbox())
            
            
            layout = timeSerWidget.verticalLayout_3 # 
            timeSerWidget.scrollArea.setWidgetResizable(True)
            timeSerWidget.scrollArea.setMinimumHeight(500)
            timeSerWidget.scrollArea.setMinimumWidth(550)
            layout.addWidget(NavigationToolbar(static_canvas, window))
            layout.addWidget(static_canvas)
    
            dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
            layout.addWidget(dynamic_canvas)
            layout.addWidget(NavigationToolbar(dynamic_canvas, window))
    
            self._static_ax = static_canvas.figure.subplots()
            # load usgs magnetic data
            
            # t = np.linspace(0, 10, 501)
            # y_t = np.tan(t)
            
            mag_t_df = api_df #get_timeseries_magnetic_data(hours=1)
            # print('assuming api_df is full', api_df)
            # print(mag_t_df)
            self.x_t = mag_t_df['time_H'].tolist()
            self.y_mag_t = mag_t_df['mag_H_nT'].tolist()
            # xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
            # self._static_ax.xaxis.set_major_formatter(xfmt)
            # static_canvas.figure.subplots_adjust(bottom=0.3)
            # self._static_ax.tick_params(rotation=15) # xticks(rotation=25)
            self._static_line, = self._static_ax.plot(self.x_t, self.y_mag_t, ".")
            
            # Add cursor tooltips for static plot
            if MPLCURSORS_AVAILABLE:
                cursor_static = mplcursors.cursor(self._static_line, hover=True)
                @cursor_static.connect("add")
                def on_add_static(sel):
                    idx = sel.target.index
                    if idx < len(self.x_t) and idx < len(self.y_mag_t):
                        time_str = pd.to_datetime(self.x_t[idx]).strftime('%Y-%m-%d %H:%M:%S') if isinstance(self.x_t[idx], (datetime, pd.Timestamp)) else str(self.x_t[idx])
                        sel.annotation.set_text(f'Time: {time_str}\nValue: {self.y_mag_t[idx]:.2f} nT')
            
            self._dynamic_ax = dynamic_canvas.figure.subplots()
            # Set up a Line2D.
            # self.xdata = np.linspace(0, 10, 101)
            # self.xdata = x_t
            # self._update_xydata(force=True)
            # self._line, = self._dynamic_ax.plot(self.xdata, self.ydata)
            self._line, = self._dynamic_ax.plot(self.x_t, self.y_mag_t)
            
            # Add cursor tooltips for dynamic plot
            if MPLCURSORS_AVAILABLE:
                cursor_dynamic = mplcursors.cursor(self._line, hover=True)
                @cursor_dynamic.connect("add")
                def on_add_dynamic(sel):
                    idx = sel.target.index
                    if idx < len(self.x_t) and idx < len(self.y_mag_t):
                        time_str = pd.to_datetime(self.x_t[idx]).strftime('%Y-%m-%d %H:%M:%S') if isinstance(self.x_t[idx], (datetime, pd.Timestamp)) else str(self.x_t[idx])
                        sel.annotation.set_text(f'Time: {time_str}\nValue: {self.y_mag_t[idx]:.2f} nT')
            
            self._save_data(self.x_t, self.y_mag_t)
            self._line_new = None
            # The below two timers must be attributes of self, so that the garbage
            # collector won't clean them after we finish with __init__...
    
            # The data retrieval may be fast as possible (Using QRunnable could be
            # even faster).
            self.data_timer = dynamic_canvas.new_timer(1000*20) # update data every 20 sec
            self.data_timer.add_callback(self._update_xydata)
            self.data_timer.start()
            # Drawing at 5 Hz should be fast enough for the GUI to feel smooth, and
            # not too fast for the GUI to be overloaded with events that need to be
            # processed while the GUI element is changed.
            self.drawing_timer = dynamic_canvas.new_timer(200)
            self.drawing_timer.add_callback(self._update_canvas)
            self.drawing_timer.start()
            self.magetic_time_ser_canvas = static_canvas
        except Exception as e:
            self.log('Error loading plot framework', error=e)
    
    def start_prediction_process(self, input_file):
        try:
            # Use the current Python interpreter (cross-platform)
            python_exe = sys.executable
            python_app_file = os.path.join(APP_BASE, 'predictor_ai.py')

            # Use session folder as working directory so relative paths resolve correctly
            work_dir = os.path.dirname(input_file) if input_file else APP_BASE
            
            # Construct command as a list for better cross-platform compatibility
            command = [python_exe, python_app_file, input_file]
            
            self.log(f'Starting prediction process: {" ".join(command)} (cwd={work_dir})', level='Info')
            self.predictor_input_file = input_file
            
            # Capture stdout/stderr to files in the session folder for debugging
            stdout_path = os.path.join(work_dir, 'predict_stdout.log')
            stderr_path = os.path.join(work_dir, 'predict_stderr.log')
            stdout_f = open(stdout_path, 'w')
            stderr_f = open(stderr_path, 'w')

            # Pass training window to predictor via environment variable if set
            env = os.environ.copy()
            if self.train_window_minutes:
                env["TRAIN_WINDOW_MINUTES"] = str(self.train_window_minutes)
            else:
                env.pop("TRAIN_WINDOW_MINUTES", None)

            self.prediction_process = subprocess.Popen(
                command,
                stdout=stdout_f,
                stderr=stderr_f,
                cwd=work_dir,  # Set working directory to session dir
                env=env
            )
            self.log(f'Prediction process started (PID: {self.prediction_process.pid})', level='Info')
        except Exception as e:
            self.log(f'Error starting prediction process: {e}', error=e, level='Error')
    
    def _update_xydata(self, force = False):
        # Shift the sinusoid as a function of time.
        # self.ydata = np.sin(self.xdata + time.time())
        
        # update logic:
        if self.new_x_t:
            # mag_t_df = #get_timeseries_magnetic_data(start_time=self.new_x_t[-1])
            start_time=self.new_x_t[-1]
        elif self.x_t:
            # mag_t_df = #get_timeseries_magnetic_data(start_time=self.x_t[-1])
            start_time=self.x_t[-1]
        
        if not force:
            self.appWin.startThreads(hours=None, start_time=start_time, new=True)
            print('after starting thread..')
        else:
            mag_t_df = api_df_new
            if mag_t_df is not None and not mag_t_df.empty and 'time_H' in mag_t_df.columns and 'mag_H_nT' in mag_t_df.columns:
                new_x_t = mag_t_df['time_H'].tolist()
                new_y_mag_t = mag_t_df['mag_H_nT'].tolist()
                if new_x_t and new_y_mag_t and len(new_x_t)>1 and len(new_y_mag_t)>1:
                    print('got new data', new_x_t[1:])
                    self.new_x_t = self.new_x_t + new_x_t[1:]
                    self.new_y_mag_t = self.new_y_mag_t + new_y_mag_t[1:]
                    self.needs_update_lims = True
            # else:
            #     print('no new data')
        self._update_predictions_data()

    def _update_predictions_data(self):
        try:
            predict_input = self.predictor_input_file
            if not predict_input:
                return  # No prediction input file set yet
            
            predict_output_file = os.path.join(os.path.dirname(predict_input), 'predict_out.csv')
            
            # Check if prediction process is still running
            if self.prediction_process:
                return_code = self.prediction_process.poll()
                if return_code is None:
                    # Process is still running
                    self.log(f'Prediction process still running (PID: {self.prediction_process.pid})', level='Debug')
                    return
                elif return_code != 0:
                    # Process finished with error
                    work_dir = os.path.dirname(predict_input)
                    stderr_log = os.path.join(work_dir, 'predict_stderr.log')
                    stderr_output = ''
                    try:
                        if os.path.exists(stderr_log):
                            with open(stderr_log, 'r') as ef:
                                stderr_output = ef.read()
                    except Exception as e_read:
                        stderr_output = f'Failed to read stderr log: {e_read}'
                    self.log(f'Prediction process failed with return code {return_code}. Error: {stderr_output}', level='Error')
                    self.prediction_process = None
                    self.predict_app_started = False
                    return
                else:
                    # Process finished successfully
                    self.log(f'Prediction process completed successfully (PID: {self.prediction_process.pid})', level='Info')
            
            if os.path.exists(predict_output_file):
                predictions = pd.read_csv(predict_output_file)
                predictions['x'] = pd.to_datetime(predictions['x'])
                __x = predictions['x'].to_list()
                __y = predictions['y'].to_list()
                if len(__x)>0:
                    # Merge with existing predictions to keep a continuous, deduped time series
                    existing_times = []
                    existing_values = []
                    if self.predict_x_t and self.predict_y_t:
                        for pt, pv in zip(self.predict_x_t, self.predict_y_t):
                            existing_times.extend(pt)
                            existing_values.extend(pv)

                    # Preserve earlier predictions: do not overwrite existing timestamps
                    merged = {}
                    for t, v in zip(existing_times, existing_values):
                        merged[t] = v
                    skipped_dupes = 0
                    for t, v in zip(__x, __y):
                        if t in merged:
                            skipped_dupes += 1
                            continue
                        merged[t] = v

                    merged_times = sorted(merged.keys())
                    merged_values = [merged[t] for t in merged_times]

                    self.predict_x_t = [merged_times]
                    self.predict_y_t = [merged_values]
                    msg = f'Loaded latest prediction: {len(__x)} points. Combined prediction series now {len(merged_times)} points.'
                    if skipped_dupes:
                        msg += f' Preserved {skipped_dupes} existing timestamps (no overwrites).'
                    self.log(msg, level='Info')
                    # Force redraw of prediction line on next canvas update
                    self.predict_last_t_in_plot = None
                    if self.prediction_process:
                        self.prediction_process = None
                        self.predict_app_started = False
                    
                    # Perform anomaly detection when we have both actual and predicted data
                    self._detect_anomalies()
                    # Trigger immediate canvas update to show new predictions
                    self._update_canvas()
            else:
                # Output file doesn't exist yet - process might still be running or failed
                if self.prediction_process and self.prediction_process.poll() is None:
                    self.log(f'Waiting for prediction output file: {predict_output_file}', level='Debug')
                else:
                    work_dir = os.path.dirname(predict_input) if predict_input else APP_BASE
                    stderr_log = os.path.join(work_dir, 'predict_stderr.log')
                    stderr_output = ''
                    try:
                        if os.path.exists(stderr_log):
                            with open(stderr_log, 'r') as ef:
                                stderr_output = ef.read()
                    except Exception as e_read:
                        stderr_output = f'Failed to read stderr log: {e_read}'
                    self.log(f'Prediction output file not found: {predict_output_file}. Process may have failed. Stderr: {stderr_output}', level='Warning')
                            
        except Exception as e:
            self.log('Error updating predictions data', error=e, level='Error')
    
    def _sync_threshold_spinbox(self):
        """Sync the threshold spinbox with the current anomaly detector value"""
        try:
            if hasattr(self, 'timeSerWidget') and hasattr(self.timeSerWidget, 'threshold_spinbox') and hasattr(self, 'anomaly_detector'):
                self.timeSerWidget.threshold_spinbox.setValue(self.anomaly_detector.threshold_multiplier)
                self.log(f'Anomaly Detection: Threshold control initialized with value {self.anomaly_detector.threshold_multiplier:.2f}', level='Debug')
            if hasattr(self, 'timeSerWidget') and hasattr(self.timeSerWidget, 'train_window_spinbox'):
                tw = getattr(self, "train_window_minutes", None)
                if tw is None:
                    self.timeSerWidget.train_window_spinbox.setValue(-1)
                elif tw == 0:
                    self.timeSerWidget.train_window_spinbox.setValue(0)
                else:
                    self.timeSerWidget.train_window_spinbox.setValue(int(tw))
        except Exception as e:
            self.log(f'Error syncing threshold spinbox: {e}', level='Error')
    
    def _detect_anomalies(self):
        """
        Detect anomalies by comparing real-time actual data with predictions.
        This method is called whenever new predictions are available.
        Only compares new real-time data (green line) with predictions (purple line), not historic data.
        """
        try:
            # Only use real-time data (new_x_t, new_y_mag_t) for anomaly detection
            # This is the green line - real-time data that should be compared with predictions
            realtime_actual_times = self.new_x_t if self.new_x_t else []
            realtime_actual_values = self.new_y_mag_t if self.new_y_mag_t else []
            
            # Get all predictions (flatten the list of lists)
            all_predicted_times = []
            all_predicted_values = []
            if self.predict_x_t and self.predict_y_t:
                for pred_times, pred_values in zip(self.predict_x_t, self.predict_y_t):
                    all_predicted_times.extend(pred_times)
                    all_predicted_values.extend(pred_values)
            
            # Only detect anomalies if we have both real-time actual and predicted data
            if len(realtime_actual_times) > 0 and len(all_predicted_times) > 0:
                self.log(f'Anomaly Detection: Starting detection with {len(realtime_actual_times)} real-time actual points and {len(all_predicted_times)} predicted points', level='Info')
                self.log(f'Anomaly Detection: Current threshold multiplier: {self.anomaly_detector.threshold_multiplier:.2f}', level='Info')
                self.log(f'Anomaly Detection: Before detection - prediction_errors count: {len(self.anomaly_detector.prediction_errors)}', level='Debug')
                
                # Detect anomalies using the AnomalyDetector
                # Only compare real-time data with predictions
                # This will recalculate the threshold using the current threshold_multiplier
                anomalies_df, threshold = self.anomaly_detector.detect_anomalies(
                    actual_times=realtime_actual_times,
                    actual_values=realtime_actual_values,
                    predicted_times=all_predicted_times,
                    predicted_values=all_predicted_values
                )
                
                # Verify threshold was calculated correctly
                expected_threshold = None
                if len(self.anomaly_detector.prediction_errors) >= self.anomaly_detector.min_samples_for_threshold:
                    mean_err = np.mean(self.anomaly_detector.prediction_errors)
                    std_err = np.std(self.anomaly_detector.prediction_errors)
                    expected_threshold = mean_err + self.anomaly_detector.threshold_multiplier * std_err
                    if abs(threshold - expected_threshold) > 0.01:
                        self.log(f'Anomaly Detection: WARNING - Threshold mismatch! Expected: {expected_threshold:.2f}, Got: {threshold:.2f}', level='Warning')
                
                self.log(f'Anomaly Detection: Detection completed. Found {len(anomalies_df)} anomalies. Threshold used: {threshold:.2f} nT (multiplier: {self.anomaly_detector.threshold_multiplier:.2f})', level='Info')
                
                # Log detailed statistics for debugging
                stats = self.anomaly_detector.get_statistics()
                
                # First, check how many matched pairs we have
                differences_df = self.anomaly_detector.calculate_differences(
                    realtime_actual_times, realtime_actual_values, all_predicted_times, all_predicted_values
                )
                
                if len(differences_df) > 0:
                    self.log(f'Anomaly Detection: Matched {len(differences_df)} pairs between actual and predicted data', level='Info')
                    max_diff = differences_df['difference'].abs().max()
                    min_diff = differences_df['difference'].abs().min()
                    mean_diff = differences_df['difference'].abs().mean()
                    std_diff = differences_df['difference'].abs().std()
                    self.log(f'Anomaly Detection: Differences - Max: {max_diff:.2f} nT, Min: {min_diff:.2f} nT, Mean: {mean_diff:.2f} nT, Std: {std_diff:.2f} nT', level='Info')
                    
                    # Show how many would be anomalies at current threshold
                    if threshold is not None:
                        potential_anomalies = (differences_df['difference'].abs() > threshold).sum()
                        self.log(f'Anomaly Detection: {potential_anomalies} out of {len(differences_df)} matched pairs exceed threshold of {threshold:.2f} nT', level='Info')
                else:
                    # No matches found - this is the problem!
                    self.log(f'Anomaly Detection: WARNING - No matched pairs found between actual and predicted data!', level='Warning')
                    if len(realtime_actual_times) > 0 and len(all_predicted_times) > 0:
                        actual_start = pd.to_datetime(realtime_actual_times[0])
                        actual_end = pd.to_datetime(realtime_actual_times[-1])
                        pred_start = pd.to_datetime(all_predicted_times[0])
                        pred_end = pd.to_datetime(all_predicted_times[-1])
                        self.log(f'Anomaly Detection: Real-time actual data range: {actual_start} to {actual_end}', level='Warning')
                        self.log(f'Anomaly Detection: Predicted data range: {pred_start} to {pred_end}', level='Warning')
                        time_gap = abs((pred_start - actual_end).total_seconds() / 60)  # gap in minutes
                        self.log(f'Anomaly Detection: Time gap between actual end and predicted start: {time_gap:.1f} minutes (tolerance: 10 minutes)', level='Warning')
                
                if stats['total_samples'] > 0:
                    self.log(f'Anomaly Detection: Statistics - Mean error: {stats["mean_error"]:.2f} nT, Std error: {stats["std_error"]:.2f} nT, Samples: {stats["total_samples"]}', level='Info')
                    self.log(f'Anomaly Detection: Current threshold multiplier: {self.anomaly_detector.threshold_multiplier:.2f}', level='Info')
                    self.log(f'Anomaly Detection: Threshold calculation - Mean + ({self.anomaly_detector.threshold_multiplier:.2f} × Std) = {stats["mean_error"]:.2f} + ({self.anomaly_detector.threshold_multiplier:.2f} × {stats["std_error"]:.2f}) = {threshold:.2f} nT', level='Info')
                    
                    # Show detailed comparison of differences vs threshold
                    if len(differences_df) > 0:
                        max_diff = differences_df['difference'].abs().max()
                        min_diff = differences_df['difference'].abs().min()
                        # Count how many exceed threshold
                        exceeding_threshold = (differences_df['difference'].abs() > threshold).sum()
                        self.log(f'Anomaly Detection: Differences vs Threshold - Max: {max_diff:.2f} nT, Min: {min_diff:.2f} nT, Threshold: {threshold:.2f} nT', level='Info')
                        self.log(f'Anomaly Detection: {exceeding_threshold} out of {len(differences_df)} differences exceed threshold', level='Info')
                        
                        if len(anomalies_df) == 0 and exceeding_threshold > 0:
                            self.log(f'Anomaly Detection: WARNING - {exceeding_threshold} differences exceed threshold but no anomalies in result! This may indicate a filtering issue.', level='Warning')
                        elif len(anomalies_df) == 0:
                            self.log(f'Anomaly Detection: No anomalies detected because all differences ({max_diff:.2f} nT max) are below threshold ({threshold:.2f} nT)', level='Info')
                else:
                    self.log(f'Anomaly Detection: Not enough samples for threshold calculation. Need {self.anomaly_detector.min_samples_for_threshold}, have {len(self.anomaly_detector.prediction_errors)}', level='Warning')
                    self.log(f'Anomaly Detection: Using default threshold: {threshold:.2f} nT (multiplier {self.anomaly_detector.threshold_multiplier:.2f} not used until enough samples)', level='Warning')
                
                # Update anomaly data for visualization
                if not anomalies_df.empty:
                    # IMPORTANT: Only show anomalies where we have BOTH actual and predicted data
                    # Filter to ensure 'predicted' column is not NaN (valid match exists)
                    # The AnomalyDetector already filtered for valid matches, so we can trust these
                    valid_anomalies = anomalies_df.dropna(subset=['predicted', 'actual'])
                    
                    if len(valid_anomalies) > 0:
                        # Since AnomalyDetector already validated matches, we can use these directly
                        new_anomaly_times = pd.to_datetime(valid_anomalies['time']).tolist()
                        new_anomaly_values = valid_anomalies['actual'].tolist()
                        
                        # FIXED: Accumulate anomalies instead of replacing (retain across cycles)
                        # Convert existing times to datetime for comparison
                        existing_times_set = set()
                        if self.anomaly_times:
                            existing_times_set = {pd.to_datetime(t) if not isinstance(t, (datetime, pd.Timestamp)) else t for t in self.anomaly_times}
                        
                        # Add new anomalies, avoiding duplicates (same timestamp)
                        new_count = 0
                        max_anomalies = 1000  # Reasonable limit to prevent memory issues
                        for new_time, new_value in zip(new_anomaly_times, new_anomaly_values):
                            new_time_dt = pd.to_datetime(new_time) if not isinstance(new_time, (datetime, pd.Timestamp)) else new_time
                            if new_time_dt not in existing_times_set:
                                # If we're at the limit, remove oldest anomaly (FIFO)
                                if len(self.anomaly_times) >= max_anomalies:
                                    self.anomaly_times.pop(0)
                                    self.anomaly_values.pop(0)
                                    # Remove from set as well
                                    if self.anomaly_times:
                                        oldest_time = pd.to_datetime(self.anomaly_times[0]) if not isinstance(self.anomaly_times[0], (datetime, pd.Timestamp)) else self.anomaly_times[0]
                                        existing_times_set.discard(oldest_time)
                                
                                self.anomaly_times.append(new_time_dt)
                                self.anomaly_values.append(new_value)
                                existing_times_set.add(new_time_dt)
                                new_count += 1
                        
                        self.log(f'Anomaly Detection: Added {new_count} new anomalies (total: {len(self.anomaly_times)}). {len(new_anomaly_times) - new_count} duplicates skipped.', level='Info')
                        if len(anomalies_df) > len(valid_anomalies):
                            self.log(f'Anomaly Detection: Filtered out {len(anomalies_df) - len(valid_anomalies)} anomalies that had NaN values', level='Debug')
                        # Force immediate canvas update to show anomalies
                        QTimer.singleShot(100, self._update_canvas)
                    else:
                        # FIXED: Don't clear existing anomalies if no new valid anomalies found
                        # Keep previous anomalies visible
                        self.log(f'Anomaly Detection: No valid anomalies found in this cycle (all had NaN values after matching). Retaining {len(self.anomaly_times)} previous anomalies.', level='Debug')
                    
                    # Log detected anomalies with detailed information
                    num_anomalies = len(anomalies_df)
                    stats = self.anomaly_detector.get_statistics()
                    max_diff = anomalies_df['difference'].abs().max()
                    min_diff = anomalies_df['difference'].abs().min()
                    avg_diff = anomalies_df['difference'].abs().mean()
                    
                    self.log(f'Anomaly Detection: {num_anomalies} anomalies detected. Threshold: {threshold:.2f} nT. Multiplier: {self.anomaly_detector.threshold_multiplier:.2f}', level='Info')
                    self.log(f'Anomaly Detection: Statistics - Mean error: {stats["mean_error"]:.2f} nT, Std error: {stats["std_error"]:.2f} nT, Samples: {stats["total_samples"]}', level='Info')
                    self.log(f'Anomaly Detection: Anomaly differences - Max: {max_diff:.2f} nT, Min: {min_diff:.2f} nT, Avg: {avg_diff:.2f} nT', level='Info')
                else:
                    # FIXED: Don't clear existing anomalies if no anomalies detected in this cycle
                    # Keep previous anomalies visible across cycles
                    stats = self.anomaly_detector.get_statistics()
                    self.log(f'Anomaly Detection: No anomalies detected in this cycle. Threshold: {threshold:.2f} nT. Mean error: {stats["mean_error"]:.2f} nT. Retaining {len(self.anomaly_times)} previous anomalies.', level='Debug')
            else:
                if len(realtime_actual_times) == 0:
                    self.log('Anomaly Detection: No real-time actual data available for comparison', level='Debug')
                if len(all_predicted_times) == 0:
                    self.log('Anomaly Detection: No predicted data available for comparison', level='Debug')
                    
        except Exception as e:
            self.log('Error detecting anomalies', error=e, level='Error')
                
    def _update_canvas(self):
        # print('updating canvas.. with new line', self._line_new)
        try:
            if not hasattr(self, '_new_predictions_line'):
                self._new_predictions_line = None
            if not self._line_new:
                if self.new_x_t and self.new_y_mag_t:
                    self._line_new, = self._dynamic_ax.plot(self.new_x_t, self.new_y_mag_t, color=[0.1, 0.7, 0.2])
                    print('green line created')
                    print('with data', self.new_x_t, self.new_y_mag_t)
                    # Add cursor tooltips for new data line
                    if MPLCURSORS_AVAILABLE:
                        cursor_new = mplcursors.cursor(self._line_new, hover=True)
                        @cursor_new.connect("add")
                        def on_add_new(sel):
                            idx = sel.target.index
                            if idx < len(self.new_x_t) and idx < len(self.new_y_mag_t):
                                time_str = pd.to_datetime(self.new_x_t[idx]).strftime('%Y-%m-%d %H:%M:%S') if isinstance(self.new_x_t[idx], (datetime, pd.Timestamp)) else str(self.new_x_t[idx])
                                sel.annotation.set_text(f'Time: {time_str}\nValue: {self.new_y_mag_t[idx]:.2f} nT')
                    self._save_data(self.x_t + self.new_x_t, self.y_mag_t + self.new_y_mag_t)
            else:
                # self._line.set_data(self.xdata, self.ydata)
                self._line_new.set_data(self.new_x_t, self.new_y_mag_t)
                self._save_data(self.x_t + self.new_x_t, self.y_mag_t + self.new_y_mag_t)
                
                if self.needs_update_lims:
                    _xrange = self.new_x_t[-1]-self.x_t[0]
                    _ymax = max(max(self.new_y_mag_t), max(self.y_mag_t))
                    _ymin = min(min(self.new_y_mag_t), min(self.y_mag_t))
                    _yrange = _ymax - _ymin
                    self._dynamic_ax.set_xlim(self.x_t[0] - 0.05*_xrange, self.new_x_t[-1] + 0.05*_xrange)
                    self._dynamic_ax.set_ylim(_ymin - 0.05*_yrange, _ymax + 0.05*_yrange)
                    self._static_ax.set_xlim(self.x_t[0] - 0.05*_xrange, self.new_x_t[-1] + 0.05*_xrange)
                    self._static_ax.set_ylim(_ymin - 0.05*_yrange, _ymax + 0.05*_yrange)
                    self.needs_update_lims = False
                # It should be safe to use the synchronous draw() method for most drawing
                # frequencies, but it is safer to use draw_idle().
                self._line.figure.canvas.draw_idle()
                self._line_new.figure.canvas.draw_idle()
                self._static_line.figure.canvas.draw_idle()
                # print('canvas updated')
            
            # Always (re)draw predictions from the merged series
            if self.predict_x_t and self.predict_y_t:
                pred_times = self.predict_x_t[-1]
                pred_values = self.predict_y_t[-1]
                if not self._new_predictions_line:
                    self._new_predictions_line, = self._dynamic_ax.plot(pred_times, pred_values, color=[0.3, 0.1, 0.4])
                    if MPLCURSORS_AVAILABLE:
                        cursor_pred = mplcursors.cursor(self._new_predictions_line, hover=True)
                        @cursor_pred.connect("add")
                        def on_add_pred(sel):
                            idx = sel.target.index
                            if idx < len(pred_times) and idx < len(pred_values):
                                time_str = pd.to_datetime(pred_times[idx]).strftime('%Y-%m-%d %H:%M:%S') if isinstance(pred_times[idx], (datetime, pd.Timestamp)) else str(pred_times[idx])
                                sel.annotation.set_text(f'Time: {time_str}\nPredicted: {pred_values[idx]:.2f} nT')
                else:
                    self._new_predictions_line.set_data(pred_times, pred_values)
                # Update last prediction endpoint
                self.predict_last_t_in_plot = pred_times[-1]
            
            # Plot anomalies if any are detected
            if self.anomaly_times and self.anomaly_values and len(self.anomaly_times) > 0:
                self.log(f'Anomaly Visualization: Attempting to plot {len(self.anomaly_times)} anomalies', level='Info')
                # Convert times to proper format if needed (handle datetime objects)
                try:
                    # Ensure times are in the right format for matplotlib
                    if isinstance(self.anomaly_times[0], (datetime, pd.Timestamp)):
                        # Times are datetime objects - matplotlib can handle them directly
                        anomaly_times_plot = self.anomaly_times
                    else:
                        # Try to convert if they're strings
                        anomaly_times_plot = [pd.to_datetime(t) if isinstance(t, str) else t for t in self.anomaly_times]
                    
                    # Debug: log first few anomaly times and values
                    if len(anomaly_times_plot) > 0:
                        self.log(f'Anomaly Visualization: First anomaly - Time: {anomaly_times_plot[0]}, Value: {self.anomaly_values[0]:.2f} nT', level='Info')
                except Exception as e:
                    self.log(f'Anomaly Visualization: Error converting times: {e}', level='Error')
                    anomaly_times_plot = self.anomaly_times
                
                # Remove old vertical lines if they exist
                for vline in self.anomaly_vertical_lines:
                    try:
                        vline.remove()
                    except:
                        pass
                self.anomaly_vertical_lines = []
                
                for vline in self.anomaly_vertical_lines_static:
                    try:
                        vline.remove()
                    except:
                        pass
                self.anomaly_vertical_lines_static = []
                
                # Get y-axis limits for vertical lines
                ylim_dynamic = self._dynamic_ax.get_ylim()
                ylim_static = self._static_ax.get_ylim()
                
                # Create vertical lines at each anomaly time point (only vertical lines, no markers)
                for anomaly_time in anomaly_times_plot:
                    # Transparent light red vertical line on dynamic plot
                    vline_dynamic = self._dynamic_ax.axvline(
                        x=anomaly_time,
                        color='lightcoral',  # Light red color
                        linestyle='-',  # Solid line instead of dashed
                        linewidth=2,
                        alpha=0.4,  # More transparent for subtle highlighting
                        zorder=6  # Above data lines but subtle
                    )
                    self.anomaly_vertical_lines.append(vline_dynamic)
                    
                    # Transparent light red vertical line on static plot
                    vline_static = self._static_ax.axvline(
                        x=anomaly_time,
                        color='lightcoral',  # Light red color
                        linestyle='-',  # Solid line instead of dashed
                        linewidth=2,
                        alpha=0.4,  # More transparent for subtle highlighting
                        zorder=6  # Above data lines but subtle
                    )
                    self.anomaly_vertical_lines_static.append(vline_static)
                
                self.log(f'Anomaly Visualization: Successfully plotted {len(self.anomaly_times)} anomaly vertical lines', level='Info')
                
                # Force redraw of both canvases
                try:
                    self._dynamic_ax.figure.canvas.draw_idle()
                    self._static_ax.figure.canvas.draw_idle()
                except Exception as e:
                    self.log(f'Anomaly Visualization: Error redrawing canvas: {e}', level='Error')
            else:
                # Clear anomalies if none detected - remove vertical lines
                for vline in self.anomaly_vertical_lines:
                    try:
                        vline.remove()
                    except:
                        pass
                self.anomaly_vertical_lines = []
                
                for vline in self.anomaly_vertical_lines_static:
                    try:
                        vline.remove()
                    except:
                        pass
                self.anomaly_vertical_lines_static = []
                
                # Clear any old scatter markers if they exist (shouldn't be created anymore)
                if self.anomaly_line:
                    try:
                        self.anomaly_line.remove()
                    except:
                        pass
                    self.anomaly_line = None
                if self.anomaly_line_static:
                    try:
                        self.anomaly_line_static.remove()
                    except:
                        pass
                    self.anomaly_line_static = None
                
                # Force redraw
                self._dynamic_ax.figure.canvas.draw_idle()
                self._static_ax.figure.canvas.draw_idle()
                
        except Exception as e:
            self.log('Error updating canvas', error=e)
                
    @property
    def appWin(self):
        return self._app_window

    def initViews(self):
        """Initializes the different views (aka tabs) in our application."""
        wnd = ApplicationWindow(self)
        # wnd.showMaximized()
        return wnd

    def load_visualization_framework(self):
        self.frame = Qt.QFrame()
        window = self.appWin
        self.vl = Qt.QVBoxLayout()
        window.tab_3.setLayout(self.vl)
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)
        self.ren = vtk.vtkRenderer()
        # todo - everything related to ren, renwin, iren, cam  .... (anything related to visualisation, and assign it to a tab manager)
        self.renwin = self.vtkWidget.GetRenderWindow()
        self.iren = self.renwin.GetInteractor()
        
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.style = style
        self.iren.SetInteractorStyle(style)
        
  #       vtkNew<InteractorStyleMoveVertex> style;
  #         renderWindowInteractor->SetInteractorStyle(style);
  #         style->Data = input;
  # style->GlyphData = glyphFilter->GetOutput();

        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        
        self.cam = self.ren.GetActiveCamera()
        
        colors = vtkNamedColors()
        self.ren.SetBackground(colors.GetColor3d('SlateGray'))
        
        # ######## TRIAL HOVER
        # colors = vtkNamedColors()
        #
        # sphere_source = vtk.vtkSphereSource()
        #
        # # Create a mapper and actor
        # mapper = vtkPolyDataMapper()
        # # sphere_source >> mapper
        # actor = vtkActor()
        # actor.SetMapper(mapper)
        # actor.GetProperty().SetColor(colors.GetColor3d('MistyRose'))
        # self.ren.AddActor(actor)
        # # Create the widget
        # from vtkmodules.vtkInteractionWidgets import vtkHoverWidget
        # hover_widget = vtkHoverWidget()
        # hover_widget.SetInteractor(self.iren)
        # hover_widget.timer_duration = 1000
        # appself = self
        # # Create a callback to listen to the widget's two VTK events.
        # class HoverCallback:
        #
        #     def __call__(self, caller, ev):
        #         if ev == 'TimerEvent':
        #             print('TimerEvent -> The mouse stopped moving and the widget hovered.')
        #         if ev == 'EndInteractionEvent':
        #             print('EndInteractionEvent -> The mouse started to move.')
        #         x, y = appself.iren.GetEventPosition()
        #         renderer = appself.iren.FindPokedRenderer(x, y)
        #         print(x, y)
        #         picker = vtk.vtkPointPicker()
        #         picker.SetTolerance(0.5)
        #         picker.Pick(x, y, 0, renderer) #appself.iren.GetRenderWindow().GetRenderers().GetFirstRenderer())
        #
        #         point_idx = picker.GetPointId()
        #         print(point_idx)
        #         if point_idx != -1:
        #             mesh = picker.GetDataSet()
        #             print('picked', mesh)
        #         # picker.Pick(x, y, 0, renderer)
        #         # point_idx = picker.GetPointId()
        #         #
        #         # if point_idx != -1:
        #         #     mesh = picker.GetDataSet()
        #         #     print(mesh.field_data['name'][0], event_name, point_idx, mesh.point_data['data'][point_idx])
        #
        # hoverCallback = HoverCallback()
        # hover_widget.AddObserver('TimerEvent', hoverCallback)
        # hover_widget.AddObserver('EndInteractionEvent', hoverCallback)
        # ##################
        
        
        
        # self.load_sphere_source()
        
        temp_blocked_for_other_analysis = True
        if not temp_blocked_for_other_analysis:
            self.load_world_grid()
            self.load_map()
        else:
            pass # code related to other temporary analysis
        
        
        
        # hover_widget.On()
        self.load_axes()
        self.ren.ResetCamera()
        self.iren.Initialize()
        self.iren.Start()
        self.ren.Render()
        self.appWin.setup_point_picker_framework()
        # self.hover_widget= hover_widget
        # hover_widget.On()
    

        
        
    def load_axes(self):
        # ########################
        axesActor1 = vtk.vtkAxesActor()
        self.axes = vtk.vtkOrientationMarkerWidget()
        self.axes.SetOrientationMarker(axesActor1)
        self.axes.SetInteractor(self.iren)
        
        
        axesActor2 = vtk.vtkAxesActor()
        axesActor2.AxisLabelsOff()
        self.ren.AddActor(axesActor2)
        # ########################
        # not so good looking so far
        # axesActor3 = vtkCubeAxesActor();
        # axesActor3.SetBounds(0, 1, 0, 1, 0, 1)
        # axesActor3.SetCamera(self.cam)
        # self.ren.AddActor(axesActor3)
        
        self.axes.EnabledOn() # <== application freeze-crash?
        self.axes.InteractiveOn()
        
        # print(self.iren.__class__, self.iren.GetInteractorStyle(), dir(self.iren))
    
    def load_sphere_source(self):
        # Create source
        source = vtk.vtkSphereSource()
        source.SetCenter(0, 0, 0)
        source.SetRadius(5.0)
        
        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        
        # # Create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.2)
        #
        self.ren.AddActor(actor)
    
    def load_world_grid(self):
        reader = load_vtk_file('salome_cube_world_res_10_with_magneticfield.vtk')
        output = reader.GetOutput()
        self.world_extent = output.GetBounds()
        # print('world extent', output.GetBounds())
        # scalar_range = output.GetScalarRange()
        # print(scalar_range)
    
        # Create the mapper that corresponds the objects of the vtk.vtk file
        # into graphics elements
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(output)
        # mapper.SetScalarRange(scalar_range)
        # mapper.ScalarVisibilityOff()
    
        # Create the Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetLineWidth(2.0)
        actor.GetProperty().SetColor(colors.GetColor3d("MistyRose"))
        actor.GetProperty().SetOpacity(0.05)
        self.ren.AddActor(actor)
        
    def load_map(self):
        # India Map
        map_figure = os.path.join(APP_BASE, *r'data\india_India_Country_Boundary_MAPOG\code_generated_Figure_1_India_cropped.png'.split('\\'))
        reader_map = vtk.vtkPNGReader()
        reader_map.SetFileName(map_figure)
        reader_map.Update()
        
        self.map_extent = reader_map.GetDataExtent()
        x_min, x_max, y_min, y_max, z_min, z_max = self.map_extent
        # print('map extent', x_min, x_max, y_min, y_max)
        
        actor_map = vtk.vtkImageActor()
        actor_map.GetMapper().SetInputConnection(reader_map.GetOutputPort())
        actor_map.GetProperty().SetOpacity(0.2)
        if z_min == z_max == 0:
            # reposition_actor(actor_map, self.map_extent, self.world_extent) # make a function to handle transformation
            transform = vtk.vtkTransform()
            transform.PostMultiply() #; // this is the key line
            w_x_min, w_x_max, w_y_min, w_y_max, w_z_in, w_z_max = self.world_extent
            transform.Scale((w_x_max-w_x_min)*1/(x_max-x_min), (w_y_max-w_y_min)*1/(y_max-y_min), 1)
            transform.Translate(-(x_min-w_x_min), -(y_min-w_y_min), 0)
            # transform.RotateZ(90.0)
            actor_map.SetUserTransform(transform)
            # if transformed correctly, set map extent to its new value:
            self.map_extent =  w_x_min, w_x_max, w_y_min, w_y_max, 0, 0
        
        map_meta = {
                'start_lat': 5.0,
                'end_lat': 40.0,
                'start_lon': 65.0,
                'end_lon': 100.0 
            }
        self.map_meta = map_meta
        self.ren.AddActor(actor_map)
    
    def get_map_xy_from_latlon(self, lat, lon):
        map_meta = self.map_meta
        map_extent = self.map_extent
        lat_min = map_meta['start_lat']
        lat_max = map_meta['end_lat']
        lon_min = map_meta['start_lon']
        lon_max = map_meta['end_lon']
        x_min, x_max, y_min, y_max, z_, z__ = map_extent
        x = x_min + (x_max - x_min)*(lon - lon_min)/(lon_max - lon_min)
        y = y_min + (y_max - y_min)*(lat - lat_min)/(lat_max - lat_min)
        return (x,y)
    
    def get_map_lat_lon_from_xy(self, x, y):
        map_meta = self.map_meta
        map_extent = self.map_extent
        lat_min = map_meta['start_lat']
        lat_max = map_meta['end_lat']
        lon_min = map_meta['start_lon']
        lon_max = map_meta['end_lon']
        x_min, x_max, y_min, y_max, z_, z__ = map_extent
        lon = lon_min + (lon_max - lon_min)*(x - x_min)/(x_max - x_min)
        lat = lat_min + (lat_max - y_min)*(y - y_min)/(y_max - y_min)
        return (lat, lon)

if __name__ == "__main__":
    app = Application([])
    sys.exit(app.exec())

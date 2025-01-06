# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Add UPX path
upx_path = r'C:\\upx-4.2.4-win64\\upx.exe'

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('models/yolov8/*.pt', 'models/yolov8'),
        ('models/fsrcnn/*.pth', 'models/fsrcnn'),
        ('assets/icons/*', 'assets/icons'),
        ('assets/fonts/*', 'assets/fonts'),
    ],
    hiddenimports=[
        'PIL._tkinter_finder',
        'sklearn.metrics',
        'sklearn.neighbors.typedefs',
        'sklearn.neighbors.quad_tree',
        'sklearn.tree._utils',
        'processing',
        'processing.image_functions',
        'processing.background_removal',
        'gui',
        'gui.main_window',
        'gui.toolbar',
        'gui.adjust_dialog',
        'gui.resize_dialog',
        'gui.segment_dialog',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ImageEditor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_dir=r'C:\\upx-4.2.4-win64',
    console=False,
    icon='assets/icons/app.ico',
    version='file_version_info.txt'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_dir=r'C:\\upx-4.2.4-win64',
    upx_exclude=[
        'vcruntime140.dll',
        'python*.dll',
        'VCRUNTIME140.dll',
        'msvcp140.dll',
        '_ssl.pyd',
        '_socket.pyd',
        'unicodedata.pyd',
    ],
    name='ImageEditor'
)

# Specify the UPX executable
import os
os.environ['UPX'] = upx_path

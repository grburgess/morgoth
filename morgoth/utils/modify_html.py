import os
import shutil

import luigi

from morgoth.utils.package_data import get_path_of_data_dir

luigi_package_dir = os.path.dirname(luigi.__file__)


def modify_index_html():
    # First restore to the original version, to make replace work
    restore_default()

    visualizer_dir = os.path.join(luigi_package_dir, 'static', 'visualiser')
    theme_css_dir = os.path.join(luigi_package_dir, 'static', 'visualiser', 'lib', 'AdminLTE', 'css')

    # Copy custom CSS to theme css dir
    custom_css = os.path.join(get_path_of_data_dir(), 'web', 'skin-mpe.min.css')
    shutil.copyfile(custom_css, os.path.join(theme_css_dir, 'skin-mpe.min.css'))

    # Create a backup of the index.html if not already done
    if not os.access(os.path.join(visualizer_dir, 'index_backup.html'), os.F_OK):
        shutil.copyfile(os.path.join(visualizer_dir, 'index.html'), os.path.join(visualizer_dir, 'index_backup.html'))

    # open file with read permissions
    with open(os.path.join(visualizer_dir, 'index.html'),'r') as f:
        filedata = f.read()

    # Modify Navbar Headline
    filedata = filedata.replace("Luigi Task Status", "<b>Morgoth</b> Pipeline")

    filedata = filedata.replace("<title>Luigi Task Visualiser</title>", "<title>Morgoth Pipeline</title>")

    # Modify theme
    filedata = filedata.replace("""<link href="lib/AdminLTE/css/skin-green-light.min.css" rel="stylesheet"/>""",
                                """<link href="lib/AdminLTE/css/skin-mpe.min.css" rel="stylesheet"/>""")

    with open(os.path.join(visualizer_dir, 'index.html'), 'w') as f:
        f.write(filedata)


def restore_default():
    visualizer_dir = os.path.join(luigi_package_dir, 'static', 'visualiser')
    theme_css_dir = os.path.join(luigi_package_dir, 'static', 'visualiser', 'lib', 'AdminLTE', 'css')

    # Restore backup of index.html if backup existing
    if os.access(os.path.join(visualizer_dir, 'index_backup.html'), os.F_OK):
        shutil.move(os.path.join(visualizer_dir, 'index_backup.html'), os.path.join(visualizer_dir, 'index.html'))

    # Delete custom skin
    if os.access(os.path.join(theme_css_dir, 'skin-mpe.min.css'), os.F_OK):
        os.remove(os.path.join(theme_css_dir, 'skin-mpe.min.css'))

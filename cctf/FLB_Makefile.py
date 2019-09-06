# 2019/09/01 Hiroyuki Ogasawara
# vim:ts=4 sw=4 et:

tool.execScript( 'DefaultSettings.py' )

#------------------------------------------------------------------------------

TENSORFLOW_ROOT= tool.findPath( '../../tensorflow', 'TENSORFLOW_ROOT' )

genv.addIncludePaths( [
        '..',
        os.path.join( TENSORFLOW_ROOT, 'include' ),
    ] )
genv.addLibPaths( [
        os.path.join( TENSORFLOW_ROOT, 'lib' ),
    ] )

#------------------------------------------------------------------------------

src_list= [
    'CCTFSystem.cpp',
    'CCTFShape.cpp',
    'CCTFLibrary.cpp',
]

TargetName= 'CCTFLibrary'

genv.setLibDir( '../lib' )
genv.setDllDir( '../lib' )

ConfigList= [ 'Debug', 'Release' ]

#------------------------------------------------------------------------------

if genv.getHostPlatform() == 'Windows':
    env= tool.createTargetEnvironment( 'Windows' )
    if env.isValid():
        src_list2= [
                "cctest.cpp",
                "ccmnist.cpp",
            ]
        env.addLibraries( [
                'tensorflow',
            ] )
        env.addCCFlags( [
                '-DNOMINMAX',
                '-DCOMPILER_MSVC',
                '-DWIN64',
                '-DWIN32_LEAN_AND_MEAN',
                '-wd4190',
            ] )
        task= tool.addLibTasks( env, 'win32', TargetName, src_list, ConfigList, env.getSupportArchList() )
        tool.addNamedTask( genv, 'build', [task] )

#------------------------------------------------------------------------------

if genv.getHostPlatform() == 'Linux':
    env= tool.createTargetEnvironment( 'Linux' )
    if env.isValid():
        src_list2= [
                #"mnisttest.cpp",
                "cctest.cpp",
                "ccmnist.cpp",
            ]
        env.addCCFlags( [
                '-frtti',
                '-std=c++14',
                '-D_GLIBCXX_USE_CXX11_ABI=0',
                '-fPIC',
                ] )
        env.addLinkFlags( [
                '-Wl,--allow-multiple-definition',
                '-Wl,--whole-archive',
                ] )
        env.addLibraries( [
                'tensorflow',
                ] )
        env.addLibPaths( [
                '/opt/rocm/lib',
            ] )
        task= tool.addLibTasks( env, 'linux', TargetName, src_list, ConfigList, env.getSupportArchList() )
        tool.addNamedTask( genv, 'build', [task] )

#------------------------------------------------------------------------------

if genv.getHostPlatform() == 'macOS':
    env= tool.createTargetEnvironment( 'macOS' )
    if env.isValid():
        env.addCCFlags( [
                '-frtti',
                '-std=c++14',
                '-D_GLIBCXX_USE_CXX11_ABI=0',
                '-fPIC',
                ] )
        env.addLinkFlags( [
                '-Wl,--allow-multiple-definition',
                '-Wl,--whole-archive',
                ] )
        task= tool.addLibTasks( env, 'macos', TargetName, src_list_macos, ConfigList, env.getSupportArchList() )
        tool.addNamedTask( genv, 'build', [task] )

#------------------------------------------------------------------------------


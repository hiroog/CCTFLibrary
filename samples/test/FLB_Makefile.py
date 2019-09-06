# 2019/09/01 Hiroyuki Ogasawara
# vim:ts=4 sw=4 et:

#------------------------------------------------------------------------------

CCTFLIBRARY_ROOT= tool.findPath( '../..', 'CCTFLIBRARY_ROOT' )
TENSORFLOW_ROOT= tool.findPath( '../../../tensorflow', 'TENSORFLOW_ROOT' )

tool.execScript( os.path.join( CCTFLIBRARY_ROOT, 'cctf/DefaultSettings.py' ) )


genv.addIncludePaths( [
        CCTFLIBRARY_ROOT,
        os.path.join( TENSORFLOW_ROOT, 'include' ),
    ] )
genv.addLibPaths( [
        os.path.join( TENSORFLOW_ROOT, 'lib' ),
    ] )
genv.addLibraries( [
        'CCTFLibrary',
        'tensorflow',
    ] )

CCTFLibRoot= os.path.join( CCTFLIBRARY_ROOT, 'lib' )
CCTFLibRoot= CCTFLIBRARY_ROOT

#------------------------------------------------------------------------------

src_list= [
    "src/main.cpp",
    "src/cctest.cpp",
    "src/ccmnist.cpp",
    "src/MNIST_Loader.cpp",
]

TargetName= 'test'

ConfigList= [ 'Debug', 'Release' ]

def makeExeName( env, src_name ):
    if src_name:
        return  env.getExeName( src_name + '_' + env.getTargetArch() + '_' + env.getConfig() )
    return  '.'

env= tool.createTargetEnvironment( genv.getHostPlatform() )
env.EXE_NAME_FUNC= makeExeName

#------------------------------------------------------------------------------

if genv.getHostPlatform() == 'Windows':
    env.addCCFlags( [
            '-DNOMINMAX',
            '-DCOMPILER_MSVC',
            '-DWIN64',
            '-DWIN32_LEAN_AND_MEAN',
            '-wd4190',
        ] )

#------------------------------------------------------------------------------

if genv.getHostPlatform() == 'Linux' or genv.getHostPlatform() == 'macOS':
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

#------------------------------------------------------------------------------

tool.addExeTasks( env, 'build', TargetName, src_list, ConfigList, [genv.getHostArch()], CCTFLibRoot )


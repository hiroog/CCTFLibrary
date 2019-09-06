# 2019/09/01 Hiroyuki Ogasawara
# vim:ts=4 sw=4 et:

module_list= [ 'cctf' ]

tool.execSubmoduleScripts( 'FLB_Makefile.py', module_list )

tool.addSubmoduleTasks( genv, 'clean', module_list )
tool.addSubmoduleTasks( genv, 'win32', module_list )
tool.addSubmoduleTasks( genv, 'linux', module_list )
tool.addSubmoduleTasks( genv, 'macos', module_list )
tool.addSubmoduleTasks( genv, 'build', module_list )


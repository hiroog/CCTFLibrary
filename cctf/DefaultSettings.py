# 2019/09/01 Hiroyuki Ogasawara
# vim:ts=4 sw=4 et:

#------------------------------------------------------------------------------

def addCleanTask( env, task_name, dir_name= os.getcwd() ):
    def command( task ):
        import  BuildUtility
        BuildUtility.RemoveTree( os.path.join( task.cwd, task.env.OUTPUT_OBJ_DIR ) )
    clean_task= env.tool.addScriptTask( env, task_name, command )
    clean_task.cwd= os.getcwd()
    return  clean_task

tool.addCleanTask= addCleanTask

#------------------------------------------------------------------------------

def addLibTasks( env, task_name, lib_name, src_list, config_list, arch_list ):
    task_list= []
    for config in config_list:
        for arch in arch_list:
            local_env= env.clone()
            local_env.setConfig( config )
            local_env.setTargetArch( arch )
            local_env.refresh()
            task= env.tool.addLibTask( local_env, lib_name, src_list )
            task_list.append( task )
    return  env.tool.addNamedTask( env, task_name, task_list )

tool.addLibTasks= addLibTasks

#------------------------------------------------------------------------------

def addExeTasks( env, task_name, exe_name, src_list, config_list, arch_list, libpath ):
    task_list= []
    for config in config_list:
        for arch in arch_list:
            local_env= env.clone()
            local_env.setConfig( config )
            local_env.setTargetArch( arch )
            local_env.addLibPaths( [local_env.getOutputPath( os.path.join( libpath, 'lib' ) )] )
            local_env.refresh()
            task= env.tool.addExeTask( local_env, exe_name, src_list )
            task_list.append( task )
    return  env.tool.addNamedTask( env, task_name, task_list )

tool.addExeTasks= addExeTasks

#------------------------------------------------------------------------------

tool.addCleanTask( genv, 'clean' )


// 2019/09/01 Hiroyuki Ogasawara
// vim:ts=4 sw=4 noet:

#ifndef	CCTFSYSTEM_H_
#define	CCTFSYSTEM_H_

#include	<stdarg.h>
#include	<stddef.h>


namespace cctf {
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

class CCSystemAPI {
public:
	virtual void*	Alloc( size_t byte_size );
	virtual void	Free( void* ptr, size_t byte_size );
	virtual void*	LoadFile( const char* path, size_t& file_size );
	virtual bool	SaveFile( const char* path, const void* buffer, size_t data_size );
	virtual void	OutputString( const char* text );
	virtual void	OutputFormat( const char* format, va_list ap );
};

class CCSystem {
	static CCSystemAPI	DefaultAllocator;
	static CCSystemAPI*	iAllocator;
public:
	static CCSystemAPI&	R()
	{
		return	*iAllocator;
	}
	static void		RegisterAPI( CCSystemAPI* allocator );
	static void*	Alloc0( size_t byte_size );
	static void		Free( void* ptr, size_t byte_size );
	static void		ZFree( void*& ptr, size_t byte_size );
	template<typename T>
	static T*	AllocByte( size_t byte_size )
	{
		return	reinterpret_cast<T*>( Alloc0( byte_size ) );
	}
	static void*	LoadFile( const char* pat, size_t& file_size );
	static bool		SaveFile( const char* pat, const void* buffer, size_t data_size );
	static void		OutputString( const char* text );
	static void		OutputFormat( const char* format, va_list ap );
	static void		Log( const char* format ... );
	static void		Print( const char* format ... );
	static void		Error( const char* format ... );
};

#define	CC_LOG( ... )	cctf::CCSystem::Log( __VA_ARGS__ )
#define	CC_ERROR( ... )	cctf::CCSystem::Error( __VA_ARGS__ )
#define	CC_PRINT( ... )	cctf::CCSystem::Print( __VA_ARGS__ )

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
}

#endif


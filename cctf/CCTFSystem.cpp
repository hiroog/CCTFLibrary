// 2019/09/01 Hiroyuki Ogasawara
// vim:ts=4 sw=4 noet:

#include	<stdio.h>
#include	<stdlib.h>
#include	<string.h>
#include	<assert.h>
#include	"CCTFSystem.h"


namespace cctf {
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

void*	CCSystemAPI::Alloc( size_t byte_size )
{
	return	malloc( byte_size );
}

void	CCSystemAPI::Free( void* ptr, size_t byte_size )
{
	free( ptr );
}

void*	CCSystemAPI::LoadFile( const char* path, size_t& file_size )
{
	assert( path != nullptr );
	file_size= 0;
#ifdef _WIN32
	FILE*	fp= nullptr;
	if( fopen_s( &fp, path, "rb" ) != 0 || !fp ){
		return	nullptr;
	}
#else
	FILE*	fp= fopen( path, "rb" );
	if( !fp ){
		return	nullptr;
	}
#endif
	fseek( fp, 0, 2 );
	file_size= ftell( fp );
	fseek( fp, 0, 0 );
	void*	buffer= Alloc( file_size );
	if( buffer ){
		fread( buffer, 1, file_size, fp );
		fclose( fp );
	}
	return	buffer;
}

bool	CCSystemAPI::SaveFile( const char* path, const void* buffer, size_t data_size )
{
	assert( path != nullptr );
	assert( buffer != nullptr );
	assert( data_size != 0 );
#ifdef _WIN32
	FILE*	fp= nullptr;
	if( fopen_s( &fp, path, "wb" ) != 0 || !fp ){
		return	false;
	}
#else
	FILE*	fp= fopen( path, "wb" );
	if( !fp ){
		return	false;
	}
#endif
	fwrite( buffer, 1, data_size, fp );
	fclose( fp );
	return	true;
}

void	CCSystemAPI::OutputString( const char* text )
{
	printf( "%s", text );
}

void	CCSystemAPI::OutputFormat( const char* format, va_list ap )
{
	vprintf( format, ap );
}


//-----------------------------------------------------------------------------

CCSystemAPI		CCSystem::DefaultAllocator;
CCSystemAPI*	CCSystem::iAllocator= &CCSystem::DefaultAllocator;

void	CCSystem::RegisterAPI( CCSystemAPI* allocator )
{
	iAllocator= allocator;
}

void*	CCSystem::Alloc0( size_t byte_size )
{
	return	iAllocator->Alloc( byte_size );
}

void	CCSystem::Free( void* ptr, size_t byte_size )
{
	iAllocator->Free( ptr, byte_size );
}

void	CCSystem::ZFree( void*& ptr, size_t byte_size )
{
	if( ptr ){
		Free( ptr, byte_size );
		ptr= nullptr;
	}
}

void*	CCSystem::LoadFile( const char* path, size_t& file_size )
{
	return	iAllocator->LoadFile( path, file_size );
}

bool	CCSystem::SaveFile( const char* path, const void* buffer, size_t data_size )
{
	return	iAllocator->SaveFile( path, buffer, data_size );
}

void	CCSystem::OutputString( const char* text )
{
	iAllocator->OutputString( text );
}

void	CCSystem::OutputFormat( const char* format, va_list ap )
{
	iAllocator->OutputFormat( format, ap );
}

void	CCSystem::Log( const char* format ... )
{
#if !defined(NDEBUG) || defined(_DEBUG)
	va_list	ap;
	va_start( ap, format );
	OutputFormat( format, ap );
	va_end( ap );
#endif
}

void	CCSystem::Print( const char* format ... )
{
	va_list	ap;
	va_start( ap, format );
	OutputFormat( format, ap );
	va_end( ap );
}

void	CCSystem::Error( const char* format ... )
{
	va_list	ap;
	va_start( ap, format );
	OutputFormat( format, ap );
	va_end( ap );
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
}


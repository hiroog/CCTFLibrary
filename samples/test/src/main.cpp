// 2019/09/01 Hiroyuki Ogasawara
// vim:ts=4 sw=4 noet:

#include	<stdio.h>
#include	<stdlib.h>
#include	<assert.h>
#include	<cctf/CCTFSystem.h>

using namespace cctf;

//-----------------------------------------------------------------------------

class MySystemAPI : public CCSystemAPI {
public:
	size_t	AllocCount= 0;
public:
	void*	Alloc( size_t byte_size ) override
	{
		AllocCount++;
		return	malloc( byte_size );
	}
	void	Free( void* ptr, size_t byte_size ) override
	{
		AllocCount--;
		free( ptr );
	}
};

//-----------------------------------------------------------------------------

extern void	cctest();
extern void	ccmnist();

int	main()
{
	static MySystemAPI	api;
	CCSystem::RegisterAPI( &api );

	cctest();
	ccmnist();

	assert( api.AllocCount == 0 );
	return	0;
}



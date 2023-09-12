%global commit0 0a92994d729ff76a58f692d3028ca1b64b145d91
%global shortcommit0 %(c=%{commit0}; echo ${c:0:7})
%global date0 20210317

%bcond_with check

Summary:        Conversion to/from half-precision floating point format
Name:           FP16
License:        MIT
Version:        1.0^git%{date0}.%{shortcommit0}
Release:        1%{?dist}

# Only a header
BuildArch:      noarch

URL:            https://github.com/Maratyszcza/%{name}
Source0:        %{url}/archive/%{commit0}/%{name}-%{shortcommit0}.tar.gz
# https://github.com/Maratyszcza/FP16/pull/24
Patch0:         0001-Remove-psimd-for-libraries-to-link.patch
Patch1:         0002-Introduce-cmake-option-FP16_USE_SYSTEM_LIBS.patch

BuildRequires: cmake
BuildRequires: gcc-c++

%if %{with check}
BuildRequires: gtest-devel
%endif

%description
Header-only library for conversion to/from half-precision floating point formats

* Supports IEEE and ARM alternative half-precision floating-point format
  *  Property converts infinities and NaNs
  *  Properly converts denormal numbers, even on systems without denormal support
* Header-only library, no installation or build required
* Compatible with C99 and C++11
* Fully covered with unit tests and microbenchmarks


%package devel

Summary:        Conversion to/from half-precision floating point format
Provides:       %{name}-static = %{version}-%{release}

%description devel
Header-only library for conversion to/from half-precision floating point formats

* Supports IEEE and ARM alternative half-precision floating-point format
  *  Property converts infinities and NaNs
  *  Properly converts denormal numbers, even on systems without denormal support
* Header-only library, no installation or build required
* Compatible with C99 and C++11
* Fully covered with unit tests and microbenchmarks

%prep
%autosetup -p1 -n %{name}-%{commit0}

%build

%cmake \
       -DFP16_USE_SYSTEM_LIBS=ON \
%if %{without check}
       -DFP16_BUILD_TESTS=OFF \
%endif
       -DFP16_BUILD_BENCHMARKS=OFF \
       
%cmake_build

%if %{with check}
%check
%ctest
%endif

%install
%cmake_install

%files devel
%license LICENSE
%doc README.md
%{_includedir}/fp16.h
%{_includedir}/fp16/
# Not needed
%exclude %{_includedir}/fp16/__init__.py
%exclude %{_includedir}/fp16/avx.py
%exclude %{_includedir}/fp16/avx2.py
%exclude %{_includedir}/fp16/psimd.h

%changelog
* Sat Sep 09 2023 Tom Rix <trix@redhat.com> - 1.0^git20210317.0a92994-1
- Initial package

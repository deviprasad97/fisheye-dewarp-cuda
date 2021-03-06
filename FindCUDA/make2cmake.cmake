

file(READ ${input_file} depend_text)

if (NOT "${depend_text}" STREQUAL "")

  # message("FOUND DEPENDS")

  string(REPLACE "\\ " " " depend_text ${depend_text})

  # This works for the nvcc -M generated dependency files.
  string(REGEX REPLACE "^.* : " "" depend_text ${depend_text})
  string(REGEX REPLACE "[ \\\\]*\n" ";" depend_text ${depend_text})

  set(dependency_list "")

  foreach(file ${depend_text})

    string(REGEX REPLACE "^ +" "" file ${file})

    # OK, now if we had a UNC path, nvcc has a tendency to only output the first '/'
    # instead of '//'.  Here we will test to see if the file exists, if it doesn't then
    # try to prepend another '/' to the path and test again.  If it still fails remove the
    # path.

    if(NOT EXISTS "${file}")
      if (EXISTS "/${file}")
        set(file "/${file}")
      else()
        if(verbose)
          message(WARNING " Removing non-existent dependency file: ${file}")
        endif()
        set(file "")
      endif()
    endif()

    # Make sure we check to see if we have a file, before asking if it is not a directory.
    # if(NOT IS_DIRECTORY "") will return TRUE.
    if(file AND NOT IS_DIRECTORY "${file}")
      # If softlinks start to matter, we should change this to REALPATH.  For now we need
      # to flatten paths, because nvcc can generate stuff like /bin/../include instead of
      # just /include.
      get_filename_component(file_absolute "${file}" ABSOLUTE)
      list(APPEND dependency_list "${file_absolute}")
    endif()

  endforeach()

else()
  # message("FOUND NO DEPENDS")
endif()

# Remove the duplicate entries and sort them.
list(REMOVE_DUPLICATES dependency_list)
list(SORT dependency_list)

foreach(file ${dependency_list})
  string(APPEND cuda_nvcc_depend " \"${file}\"\n")
endforeach()

file(WRITE ${output_file} "# Generated by: make2cmake.cmake\nSET(CUDA_NVCC_DEPEND\n ${cuda_nvcc_depend})\n\n")

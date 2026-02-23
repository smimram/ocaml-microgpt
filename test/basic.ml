open Autograd

(* Basic autograd test. *)
let () =
  let open Infix in
  let a = const 2. in
  let b = const 3. in
  let c = a * b in
  let l = c + a in
  backward l;
  assert (grad a = 4.);
  assert (grad b = 2.)
